#include "PCA_Downsampling/voxel_grid_downsampling.h"

int main(int argc, char** argv) {
    std::string pcd_file(argv[1]);
    std::string pcd_downsample(argv[2]);

    VoxelGridDownsampling vgd;
    vgd.set_data(pcd_file);
    vgd.downsampling(0.5);
    vgd.save_to_file(pcd_downsample);
    return 0;
}