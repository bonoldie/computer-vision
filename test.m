close all;
clear all;
clc;


xmp_pose_R = [
    -0.0376026577346,-0.11992473137,0.99207061187;
   -0.987749875042, -0.145998407495, -0.0550876516489;
   0.151447101281, -0.981989065018, -0.112965710278;
];

xmp_pose_t = [-6.26692379871 -0.986993936906 6.41601378017];

cv2_pose_R = [
    -0.99687418, -0.06581572, -0.04370536;
    0.037946, 0.08634929,-0.99554201;
    0.06929624,-0.99408857, -0.08358194;
];

cv2_pose_t = [-1.84355012, 6.90487024 6.58687949];

pc_orig = pcread('downloads/dante_rework/SamPointCloud.ply');

pc = pcread('downloads/dante_rework/SamPointCloud.ply');
pc_cv2 = pcread('downloads/dante_rework/SamPointCloud.ply');

pc.Color = repmat([1, 0, 0], pc.Count, 1);
pc_cv2.Color = repmat([0, 1, 0], pc_cv2.Count, 1);

xmp_pose_transform = rigidtform3d(xmp_pose_R,xmp_pose_t);
cv2_pose_transform = rigidtform3d(fixRotationMatrix(cv2_pose_R),cv2_pose_t);

pc_xmp = pctransform(pc,xmp_pose_transform);
pc_cv2 = pctransform(pc_cv2,cv2_pose_transform);

pcshow(pc_orig)
hold on
pcshow(pc_xmp, ViewPlane="XY")
pcshow(pc_cv2, ViewPlane="XY")



function R_fixed = fixRotationMatrix(R)
    [U, ~, V] = svd(R);
    R_fixed = U * V';
    if det(R_fixed) < 0
        U(:,3) = -U(:,3);
        R_fixed = U * V';
    end
end