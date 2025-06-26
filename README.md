sTILCs
===
The codes process prediction results from TILScout to generate hotspot maps of TIL-positive patches for each WSI. They then extract features from these hotspot maps using a trained Autoencoder, perform PCA for further dimensionality reduction, and apply KMeans clustering to group the WSIs based on TIL distribution patterns. The final output includes labeled datasets for downstream analysis.

Please see TILScout via: https://github.com/huibozh/TILScout.
