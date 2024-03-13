%% brt comparisons w/ 500 train samples
clear;close all; clc;
fig_0std = openfig('figs/brt_data1h_0std.fig');
fig_3std = openfig('figs/brt_data1h_3std.fig');
fig_gt = openfig('figs/brt_gt.fig');

fig_gt_children = allchild(get(fig_gt, "CurrentAxes"));
contour_gt = fig_gt_children(1);
% contour_target_set = fig_gt_children(2);
set(contour_gt, 'EdgeColor','g');
set(contour_gt, 'FaceColor','g');
set(contour_gt, 'FaceAlpha',0.1);
% set(contour_gt, 'LineStyle','-.');
set(contour_gt, 'LineWidth',2);

fig_0std_children = allchild(get(fig_0std, "CurrentAxes"));
contour_0std = fig_0std_children(1);
set(contour_0std, 'EdgeColor','red');


fig_3std_children = allchild(get(fig_3std, "CurrentAxes"));
contour_3std = fig_3std_children(1);
set(contour_3std, 'EdgeColor','blue');

figure; grid on;
axes_composite = gca;
% axis equal;
xlim([-pi, pi]);
xlabel('\theta');
ylabel('$\dot{\theta}$', 'Interpreter','latex');
copyobj(contour_gt, axes_composite);
copyobj(contour_3std, axes_composite);
copyobj(contour_0std, axes_composite);
p1 = patch([-pi -0.6*pi -0.6*pi -pi],[-10 -10 10  10],'r','FaceAlpha',0.1, 'EdgeColor','none');
p2 = patch([0.6*pi pi pi 0.6*pi],[-10 -10 10  10],'r','FaceAlpha',0.1, 'EdgeColor','none');
legend('Ground Truth', 'Our Method', 'Baseline 1 Mean Dynamics', 'Target Set', 'Interpreter', 'latex');



