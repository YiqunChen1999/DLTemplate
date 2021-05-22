

from alphaconfig import AlphaConfig

loss_fn = AlphaConfig()

# ========      MSELoss         ========
loss_fn.MSELoss.WEIGHT1                     =   1.0
loss_fn.MSELoss.WEIGHT2                     =   1.0
# ========      MAELoss         ========
loss_fn.MAELoss.WEIGHT1                     =   1.0
loss_fn.MAELoss.WEIGHT2                     =   1.0
# ========      MSESSIMLoss     ========
loss_fn.MSESSIMLoss.mse                     =   1.0
loss_fn.MSESSIMLoss.ssim                    =   0.5