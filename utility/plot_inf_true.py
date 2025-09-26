import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

trained = 'L30P250V75H110LH40X1000Y1000O30_trained.npy'
true = 'L30P250V75H110LH40X1000Y1000O30_dumps_hist.npy'
test = 'L30P250V75H110LH40X1000Y1000O30.npy'

data_trained = np.load(trained)
data_true = np.load(true)
data_test = np.load(test)

data_true = data_true/np.sum(data_true[0,0,0,:])

#vmax_val = np.max([np.max(data_true[:,:,:,2:]),np.max(data_test[:,:,:,2:]),np.max(data_trained[:,:,:,2:])])

vmax_val = np.max([np.max(data_true[:,:,:,2:]),np.max(data_trained[:,:,:,2:])])

# for i in range(50):
    # vmax_val = np.max([np.max(data_true[:,:,:,i]),np.max(data_trained[:,:,:,i])])
    # fig, axs = plt.subplots(1,2, figsize=(17,10), sharey=True)
    # if i == 49:
        # fig.suptitle('Comparison of Predicted Grain Size Probabilities\nBefore And After Training ('+str(i*2+6)+'+ $\mu$m diameter)',fontsize=18)
    # else:
        # fig.suptitle('Comparison of Predicted Grain Size Probabilities\nBefore And After Training ('+str(i*2+6)+'-'+str((i+1)*2+6)+'$\mu$m diameter)',fontsize=18)
    
    # im0 = axs[0].imshow(data_trained[90,:,:,i].T, origin='lower', vmax=vmax_val, vmin=0.0)
    # axs[0].set_title('Prediction Before Additional Training',fontsize=14)
    # axs[0].axis('off')
    # divider0 = make_axes_locatable(axs[0])
    # cax0 = divider0.append_axes('right',size='5%',pad=0.05)
    # fig.colorbar(im0,cax=cax0,orientation='vertical')
    # plt.tight_layout()
    
    # im1 = axs[1].imshow(data_true[90,:,:,i].T, origin='lower', vmax=vmax_val, vmin=0.0)
    # axs[1].set_title('Data From A 10 Run Ensemble',fontsize=14)
    # axs[1].axis('off')
    # divider1 = make_axes_locatable(axs[1])
    # cax1 = divider1.append_axes('right',size='5%',pad=0.05)
    # fig.colorbar(im1,cax=cax1,orientation='vertical')
    # plt.tight_layout()
    
    # # im2 = axs[2].imshow(data_trained[130,:,:,i].T, origin='lower', vmax=vmax_val, vmin=0.0)
    # # axs[2].set_title('Prediction After Additional Training',fontsize=14)
    # # axs[2].axis('off')
    # # divider2 = make_axes_locatable(axs[2])
    # # cax2 = divider2.append_axes('right',size='5%',pad=0.05)
    # # fig.colorbar(im2,cax=cax2,orientation='vertical')
    # # #fig.colorbar(im2,ax=axs.ravel().tolist(),orientation='vertical')
    # # plt.tight_layout()
    
    # plt.savefig('plots/L50i'+str(i)+'.png', format='png')
    # plt.close()
    
for i in range(50):
    vmax_val = np.max([np.max(data_true[130,:,:,i]),np.max(data_test[130,:,:,i]),np.max(data_trained[130,:,:,i])])
    fig, axs = plt.subplots(1,3, figsize=(17,6), sharey=True)
    if i == 49:
        fig.suptitle('Comparison of Predicted Grain Size Probabilities\nBefore And After Training ('+str(i*2+6)+'+ $\mu$m diameter)',fontsize=18)
    else:
        fig.suptitle('Comparison of Predicted Grain Size Probabilities\nBefore And After Training ('+str(i*2+6)+'-'+str((i+1)*2+6)+'$\mu$m diameter)',fontsize=18)
    
    im0 = axs[0].imshow(data_test[130,:,:,i].T, origin='lower', vmax=vmax_val, vmin=0.0)
    axs[0].set_title('Prediction Before Additional Training',fontsize=14)
    axs[0].axis('off')
    divider0 = make_axes_locatable(axs[0])
    #cax0 = divider0.append_axes('right',size='5%',pad=0.05)
    #fig.colorbar(im0,cax=cax0,orientation='vertical')
    plt.tight_layout()
    
    im1 = axs[1].imshow(data_true[130,:,:,i].T, origin='lower', vmax=vmax_val, vmin=0.0)
    axs[1].set_title('Data From A 10 Run Ensemble',fontsize=14)
    axs[1].axis('off')
    divider1 = make_axes_locatable(axs[1])
    #cax1 = divider1.append_axes('right',size='5%',pad=0.05)
    #fig.colorbar(im1,cax=cax1,orientation='vertical')
    plt.tight_layout()
    
    im2 = axs[2].imshow(data_trained[130,:,:,i].T, origin='lower', vmax=vmax_val, vmin=0.0)
    axs[2].set_title('Prediction After Additional Training',fontsize=14)
    axs[2].axis('off')
    divider2 = make_axes_locatable(axs[2])
    cax2 = divider2.append_axes('right',size='5%',pad=0.05)
    fig.colorbar(im2,cax=cax2,orientation='vertical')
    #fig.colorbar(im2,ax=axs.ravel().tolist(),orientation='vertical')
    plt.tight_layout()
    
    plt.savefig('plots/L30i'+str(i)+'.png', format='png')
    plt.close()
    