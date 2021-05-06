#!/usr/bin/env python
# coding: utf-8

import torch
import torch.optim as optim

def norm(x):
    return (x - x.min()) / (x.max() - x.min())

def add_gaussian_noise(x, mean, std):
    return x + mean + torch.randn(x.size()) * std

def get_saliency(x, label, model):
    (model.cuda()).eval()

    # declare input data require gradient
    x.requires_grad_()

    predict = model(x.cuda())
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(predict, label.cuda())
    loss.backward()

    # absolute gradient
    saliencies = x.grad.abs().detach().cpu()
    saliencies = torch.stack([norm(img_grad) for img_grad in saliencies])
    return saliencies


layer_activations = None
def filter_explaination(x, model, layer, filterid, iteration=100, lr=1):
  # x: 要用來觀察哪些位置可以 activate 被指定 filter 的圖片們
  # layer, filterid: 想要指定第幾層 layer 中第幾個 filter
  model.eval()

  def hook(model, input, output):
    global layer_activations
    layer_activations = output
  
  hook_handler = layer.register_forward_hook(hook)

  # Filter activation: 我們先觀察 x 經過被指定 filter 的 activation map
  model(x.cuda())
  # 這行才是正式執行 forward，因為我們只在意 activation map，所以這邊不需要把 loss 存起來
  filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
  # 根據 function argument 指定的 filterid 把特定 filter 的 activation map 取出來
  # 因為目前這個 activation map 我們只是要把他畫出來，所以可以直接 detach from graph 並存成 cpu tensor
  
  
  # Filter visualization: 接著我們要找出可以最大程度 activate 該 filter 的圖片
  # 從一張 random noise 的圖片開始找 (也可以從一張 dataset image 開始找)
  x = add_gaussian_noise(x, 0.5, 0.1)
  x.requires_grad_()
  # 我們要對 input image 算偏微分
  optimizer = optim.Adam([x], lr=lr)
  # 利用偏微分和 optimizer，逐步修改 input image 來讓 filter activation 越來越大
  for iter in range(iteration):
    optimizer.zero_grad()
    model(x.cuda())
    
    objective = -layer_activations[:, filterid, :, :].sum()
    # 與上一個作業不同的是，我們並不想知道 image 的微量變化會怎樣影響 final loss
    # 我們想知道的是，image 的微量變化會怎樣影響 activation 的程度
    # 因此 objective 是 filter activation 的加總，然後加負號代表我們想要做 maximization
    
    objective.backward()
    # 計算 filter activation 對 input image 的偏微分
    optimizer.step()
    # 修改 input image 來最大化 filter activation
  filter_visualization = x.detach().cpu().squeeze()
  # 完成圖片修改，只剩下要畫出來，因此可以直接 detach 並轉成 cpu tensor

  hook_handler.remove()
  # 很重要：一旦對 model register hook，該 hook 就一直存在。如果之後繼續 register 更多 hook
  # 那 model 一次 forward 要做的事情就越來越多，甚至其行為模式會超出你預期 (因為你忘記哪邊有用不到的 hook 了)
  # 因此事情做完了之後，就把這個 hook 拿掉，下次想要再做事時再 register 就好了。

  return filter_activations, filter_visualization