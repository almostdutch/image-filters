# image-filters
A whole arsenal of image processing filters

The following filters have been implemented:<br/>
* HistEqualization

* LocalHistEqualization

* SpatiallyAdaptiveSmoothingFilter (preserves edges):
  * flat kernel (application: uniform and Gaussian noise)
  * gaussian kernel (application: uniform and Gaussian noise)
  
* OrderStatisticFilter
  * Median kernel (application: impulsive noise)
  * Min kernel (application: salt noise)
  * Max kernel (application: pepper noise)
  * Mid-point kernel (application: uniform and Gaussian noise)
  * Alpha-trimmed-mean kernel (application: Gaussian + impulsive noise)
  
* SpatiallyAdaptiveOrderStatisticFilter (preserves edges):
  * Median kernel (application: impulsive noise)
  * Min kernel (application: salt noise)
  * Max kernel (application: pepper noise)
  * Mid-point kernel (application: uniform and Gaussian noise)
  * Alpha-trimmed-mean kernel (application: Gaussian + impulsive noise)
  
* HighBoostFilter

* HomomorphicFilter
