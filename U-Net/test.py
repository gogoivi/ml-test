from overnight import load_napari_volumes
import napari

# For each experiment:
napari_dir = r'C:\Users\gogoi\Desktop\ml-test\overnight_results\comparison_20261005_091400/20261005_091403_CE_Dice_clDice_Topo/napari'
volumes = load_napari_volumes(napari_dir)

viewer = napari.Viewer()
# viewer.add_image(volumes['sample_0_image'], name='MRI')
# viewer.add_labels(volumes['sample_0_target'].astype(int), name='Ground Truth')
# viewer.add_labels(volumes['sample_0_pred'].astype(int), name='Prediction')
# napari.run()
viewer.add_image(volumes['sample_1_image'], name='MRI')
viewer.add_labels(volumes['sample_1_target'].astype(int), name='Ground Truth')
viewer.add_labels(volumes['sample_1_pred'].astype(int), name='Prediction')
napari.run()