import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .encoder import Res101Encoder
import numpy as np
import random
import cv2
from torch.nn import Softmax

class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3", alpha=0.9):
        super().__init__()

        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.criterion = nn.NLLLoss()
        self.alpha = torch.Tensor([alpha, 1 - alpha])
        self.fg_num = 100
        self.mlp1 = MLP(256, self.fg_num)
        self.attention = SCCA(inplanes=512,planes=512,kernel_size=3)
        self.attention1 = SCCA(inplanes=256, planes=256, kernel_size=3)

        self.edge_network = EdgeDetectionNetwork(in_channels=3)
        self.edge_loss_fn = nn.BCELoss()
        self.edge_loss_weight = 1.0
        self.criterion_MSE = nn.MSELoss()

    def forward(self, supp_imgs, supp_mask, qry_imgs, train=False, n_iters=0):
        self.n_ways = len(supp_imgs)
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]
        supp_bs = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)

        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view(
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        qry_fts = [img_fts[dic][self.n_ways * self.n_shots * supp_bs:].view(
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        self.t = tao[self.n_ways * self.n_shots * supp_bs:]
        self.thresh_pred = [self.t for _ in range(self.n_ways)]

        align_loss = torch.zeros(1).to(self.device)
        outputs = []
        edge_loss = 0.0
        for epi in range(supp_bs):
            fg_protos = [[[self.get_protos(supp_fts[n][[epi], way, shot], supp_mask[[epi], way, shot])
                               for shot in range(self.n_shots)] for way in range(self.n_ways)] for n in
                             range(len(supp_fts))]
            fg_proto = []
            for n in range(len(supp_fts)):
                fp = self.ag_prototypes(fg_protos[n])
                center = torch.mean(fp[0], dim=0, keepdim=True)
                similarities = torch.nn.functional.cosine_similarity(fp[0], center, dim=1)
                weights = torch.softmax(similarities, dim=0)
                fp[0] = torch.sum(fp[0] * weights.unsqueeze(1), dim=0, keepdim=True)
                fg_proto.append(fp)
            fg_pred = [torch.stack(
                [self.getPred(qry_fts[n][epi], fg_proto[n][way], self.thresh_pred[way])
                 for way in range(self.n_ways)], dim=1) for n in range(len(qry_fts))]
            fg_prototypes_ = []
            if (not train) and n_iters > 0:
                for n in range(len(qry_fts)):
                    fg_prototypes_.append(
                        self.updatePrototype(qry_fts[n], fg_proto[n], fg_pred[n], n_iters, epi))
                fg_pred = [torch.stack(
                    [self.getPred(qry_fts[n][epi], fg_prototypes_[n][way], self.thresh_pred[way]) for way in
                     range(self.n_ways)], dim=1) for n in range(len(qry_fts))]
            fg_pred_up = [F.interpolate(fg_pred[n], size=img_size, mode='bilinear', align_corners=True)
                          for n in range(len(qry_fts))]
            pred_fg = [self.alpha[n] * fg_pred_up[n] for n in range(len(qry_fts))]
            preds_fg = torch.sum(torch.stack(pred_fg, dim=0), dim=0) / torch.sum(self.alpha)
            preds = torch.cat((1.0 - preds_fg, preds_fg), dim=1)
            outputs.append(preds)
            if train:
                align_loss_epi = self.alignLoss([supp_fts[n][epi] for n in range(len(supp_fts))],
                                                [qry_fts[n][epi] for n in range(len(qry_fts))],
                                                preds, supp_mask[epi])
                align_loss += align_loss_epi
                pred_edges = self.edge_network(torch.cat([preds[:, 1:2]] * 3, dim=1))
                qry_edge_gt = self.generate_edge_gt(supp_mask[epi][0])
                edge_loss += self.edge_loss_fn(pred_edges, qry_edge_gt.unsqueeze(1))
        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])

        return output, (align_loss / supp_bs) + self.edge_loss_weight * edge_loss

    @staticmethod
    def generate_edge_gt(qry_img):
        if qry_img.shape[1] == 3:
            qry_img = qry_img.mean(dim=1, keepdim=True)
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=qry_img.dtype, device=qry_img.device)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=qry_img.dtype, device=qry_img.device)

        edge_x = F.conv2d(qry_img, sobel_x, padding=1)
        edge_y = F.conv2d(qry_img, sobel_y, padding=1)

        edge_gt = torch.sqrt(edge_x ** 2 + edge_y ** 2)
        edge_gt = torch.clamp(edge_gt, 0, 1)
        return edge_gt

    def updatePrototype(self, fts, prototype, pred, update_iters, epi):
        prototype_ = Parameter(torch.stack(prototype, dim=0))
        optimizer = torch.optim.Adam([prototype_], lr=0.01)
        while update_iters > 0:
            with torch.enable_grad():
                pred_mask = torch.sum(pred, dim=-3)
                pred_mask = torch.stack((1.0 - pred_mask, pred_mask), dim=1).argmax(dim=1, keepdim=True)
                pred_mask = pred_mask.repeat([*fts.shape[1:-2], 1, 1])
                bg_fts = fts[epi] * (1 - pred_mask)
                fg_fts = torch.zeros_like(fts[epi])
                for way in range(self.n_ways):
                    fg_fts += prototype_[way].unsqueeze(-1).unsqueeze(-1).repeat(*pred.shape) \
                              * pred_mask[way][None, ...]
                new_fts = bg_fts + fg_fts
                fts_norm = torch.sigmoid((fts[epi] - fts[epi].min()) / (fts[epi].max() - fts[epi].min()))
                new_fts_norm = torch.sigmoid((new_fts - new_fts.min()) / (new_fts.max() - new_fts.min()))
                bce_loss = nn.BCELoss()
                loss = bce_loss(fts_norm, new_fts_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.stack([self.getPred(fts[epi], prototype_[way], self.thresh_pred[way])
                                for way in range(self.n_ways)], dim=1)  # N x Wa x H' x W'
            update_iters += -1

        return prototype_

    def getPred(self, fts, prototype, thresh):
        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))

        return pred


    def getFeatures(self, fts, mask):
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C
        return masked_fts

    def getPrototype(self, fg_fts):
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]
        return fg_prototypes

    def alignLoss(self, supp_fts, qry_fts, pred, fore_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])
        pred_mask = pred.argmax(dim=1, keepdim=True).squeeze(1)
        binary_masks = [pred_mask == i for i in range(1 + n_ways + 1)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=0).float()  # (1 + Wa) x N x H' x W'
        loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            for shot in range(n_shots):
                qry_fts_ = [[self.getFeatures(qry_fts[n], pred_mask[way + 1])] for n in range(len(qry_fts))]
                fg_prototypes = [self.getPrototype([qry_fts_[n]]) for n in range(len(supp_fts))]

                supp_pred = [self.getPred(supp_fts[n][way, [shot]], fg_prototypes[n][way], self.thresh_pred[way])
                             for n in range(len(supp_fts))]  # N x Wa x H' x W'
                supp_pred = [F.interpolate(supp_pred[n][None, ...], size=fore_mask.shape[-2:], mode='bilinear',
                                           align_corners=True)
                             for n in range(len(supp_fts))]

                preds = [self.alpha[n] * supp_pred[n] for n in range(len(supp_fts))]
                preds = torch.sum(torch.stack(preds, dim=0), dim=0) / torch.sum(self.alpha)
                pred_ups = torch.cat((1.0 - preds, preds), dim=1)

                supp_label = torch.full_like(fore_mask[way, shot], 255, device=fore_mask.device)
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[fore_mask[way, shot] == 0] = 0

                eps = torch.finfo(torch.float32).eps
                log_prob = torch.log(torch.clamp(pred_ups, eps, 1 - eps))
                loss += self.criterion(log_prob, supp_label[None, ...].long()) / n_shots / n_ways

        return loss
    def ag_prototypes(self, fg_protos):
        device = next(self.parameters()).device
        attn_layer = LightweightAttention(512).to(device)
        fg_protos = [
            [shot.to(device) for shot in way]
            for way in fg_protos
        ]
        return [attn_layer(way) for way in fg_protos]

    def get_protos(self, features, mask):
        features = F.interpolate(features, size=mask.shape[-2:], mode='bilinear', align_corners=True)
        kernel_size = max(3, int(min(mask.shape[-2:]) * 0.01))
        ie_mask = mask.squeeze(0) - torch.tensor(
            cv2.erode(mask.squeeze(0).cpu().numpy(), np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=2)
        ).to(self.device).unsqueeze(0)
        weights = torch.exp(-torch.norm(features, dim=1))
        ie_prototype = torch.sum(features * ie_mask[None, ...] * weights, dim=(-2, -1)) / \
                       (ie_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        origin_prototype = torch.sum(features * mask[None, ...], dim=(-2, -1)) / \
                           (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)

        fg_features = self.get_fg_fts(features, mask)
        fg_features_pro = self.attention(fg_features)
        fg_prototypes = self.mlp1(fg_features_pro.reshape(features.size(1), -1)).permute(1, 0)

        multi_scale_prototypes = []
        for scale in [0.5, 1.0, 2.0]:
            scaled_fts = F.interpolate(fg_features_pro, scale_factor=scale, mode='bilinear', align_corners=True)
            scaled_mask = F.interpolate(mask[None, ...].float(), size=scaled_fts.shape[-2:], mode='bilinear',
                                        align_corners=True)
            scaled_mask = (scaled_mask > 0.5).float()
            scaled_prototype = torch.sum(scaled_fts * scaled_mask, dim=(-2, -1)) / \
                               (scaled_mask.sum(dim=(-2, -1)) + 1e-5)
            multi_scale_prototypes.append(scaled_prototype)
        multi_scale_prototypes = torch.stack(multi_scale_prototypes, dim=0)

        fg_protos = torch.cat([fg_prototypes, origin_prototype, ie_prototype, *multi_scale_prototypes], dim=0)
        return fg_protos

    def get_random_pts(self, features, mask, n_protptype):

        features = features.squeeze(0)
        features = features.permute(1, 2, 0)
        features = features.view(features.shape[-2] * features.shape[-3],
                                             features.shape[-1])
        mask = mask.squeeze(0).view(-1)
        indx = mask == 1
        features = features[indx]
        if len(features) >= n_protptype:
            k = random.sample(range(len(features)), n_protptype)
            prototypes = features[k]
        else:
            if len(features) == 0:
                prototypes = torch.zeros(n_protptype, 512).to(self.device)
            else:
                r = (n_protptype) // len(features)
                k = random.sample(range(len(features)), (n_protptype - len(features)) % len(features))
                prototypes = torch.cat([features for _ in range(r)], dim=0)
                prototypes = torch.cat([features[k], prototypes], dim=0)

        return prototypes

    def get_fg_fts(self, fts, mask):
        _, c, h, w = fts.shape
        fg_fts = fts * mask[None, ...]
        bg_fts = torch.ones_like(fts) * mask[None, ...]
        mask_ = mask.view(-1)
        n_pts = len(mask_) - len(mask_[mask_ == 1])
        select_pts = self.get_random_pts(fts, mask, n_pts)
        index = bg_fts == 0
        fg_fts[index] = select_pts.permute(1, 0).reshape(512 * n_pts)

        return fg_fts

class EdgeDetectionNetwork(nn.Module):
    def __init__(self, in_channels):
        super(EdgeDetectionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        edges = torch.sigmoid(self.conv2(x))
        return edges

class MLP(nn.Module):
    def __init__(self, in_dim=256, out_dim=100):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduce = nn.Sequential(
            nn.Linear(in_features=self.in_dim * self.in_dim, out_features=2048, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=2048, out_features=1024, bias=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1024, out_features=self.out_dim, bias=True)
        )

    def forward(self, x):
        out = self.reduce(x)
        return out


class LightweightAttention(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.attn = nn.Linear(feat_dim, 1)

    def forward(self, features):
        expected_device = self.attn.weight.device
        features = [f.to(expected_device) for f in features]
        stacked_features = torch.stack(features)
        weights = torch.softmax(self.attn(stacked_features), dim=0)
        return torch.sum(stacked_features * weights, dim=0)

class SCCA(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(SCCA, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)
        self.softmax_left = nn.Softmax(dim=2)
        self.cross = CrossAttention(planes)
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')
        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True
        self.conv_q_left.inited = True
        self.conv_v_left.inited = True

    def spatial_weighted(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()
        input_x = input_x.view(batch, channel, height * width)
        context_mask = self.conv_q_right(x)
        context_mask = context_mask.view(batch, 1, height * width)
        context_mask = self.softmax_right(context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        context = context.unsqueeze(-1)
        context = self.conv_up(context)
        mask_ch = self.sigmoid(context)
        out = x * mask_ch
        return out

    def channel_weighted(self, x):
        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        context = torch.matmul(avg_x, theta_x)
        context = self.softmax_left(context)
        context = context.view(batch, 1, height, width)
        mask_sp = self.sigmoid(context)
        out = x * mask_sp
        return out

    def forward(self, x):
        context_channel = self.spatial_weighted(x)
        context_channel = self.cross(context_channel)
        context_spatial = self.channel_weighted(x)
        context_spatial = self.cross(context_spatial)
        out = context_spatial + context_channel
        out = self.cross(out)
        return out

class CrossAttention(nn.Module):

    def __init__(self, in_dim):
        super(CrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width, x.device)).view(m_batchsize, width,
                                                                                                     height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)

        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def INF(B, H, W, device):
    return -torch.diag(torch.tensor(float("inf"), device=device).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)
