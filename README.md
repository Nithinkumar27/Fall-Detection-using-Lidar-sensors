# Fall detection 

In an aging global population, fall incidents pose a significant risk to the elderly, often resulting in severe consequences. Sudden falls can lead to mental and physical damage to the body, especially in elders, where it can even prove fatal. In this scenario, it becomes vital to detect falls in order to provide instant relief and assistance by notifying the required people. While the initial proposal would be to install cameras in required areas such as old age homes, which would constantly monitor the area for any falls occurring, this poses a privacy issue in high-risk regions such as bathrooms and toilets. Keeping this in mind, this paper proposes the use of LiDAR sensors that do not cause any invasion of privacy for fall detection. The Velodyne VLP-16 Lidar sensor, known for its 100 m range, compact form factor, and 905nm technology, was used for the manual data collection process. The collection of data involved the team performing various poses applicable to the scenarios in which the fall detection model would be used. This collected data was then manually annotated using the Point Processing Toolkit to categorize it into various classes.


#Dataset
Data was collected with the help of the Velodyne VLP-16 puck sensor which is a state-of-the-art sensor known for its 100 m range with compact form factor and 905nm tech. A total of 12 scenes were collected by the team where actors perfomed various day-to-day scenes applicable the use case and the lidar sensor would record and produce appropriate point clouds. The dataset provided a total of 727 individual frames which contained the point cloud data. However, the details of the dataset are presented in the later sections.

# Deep Learning Models
There are three deep learning models which we try to classification and segmentation with our own dataset and do the comparitive study on the three models.
