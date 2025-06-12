sTILCs

This code processes prediction results from TILScout to generate hotspot maps of TIL-positive patches for each WSI. It then extracts features from these hotspot maps using a trained Autoencoder, performs PCA for further dimensionality reduction, and applies KMeans clustering to group the WSIs based on TIL distribution patterns. The final output includes labeled datasets for downstream analysis.
