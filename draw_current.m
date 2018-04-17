function make_current_plots = draw(path_x,path_y,path_current_x,path_current_y,name_quiver,name_slice,name_stream)
% Creates three different types of plots: quiver, slice and stream for current

%Gets data from files
[x,delimiterOut]=importdata(path_x);
[y,delimiterOut]=importdata(path_y);
[current_x,delimiterOut]=importdata(path_current_x);
[current_y,delimiterOut]=importdata(path_current_y);

%Takes starting positions for plotting
starty=y;
startx1 = max(x)*ones(size(starty));
% Makes quiver plot
fig_quiver=figure;
quiver(x,y,current_x,current_y);
axis([min(x),max(x),min(y),max(y)]);
saveas(gcf,name_quiver);
saveas(gcf,name_quiver,'png');
close(fig_quiver);
% Makes slice plot
fig_slice=figure;
streamslice(x,y,current_x,current_y,'nearest');
axis([min(x),max(x),min(y),max(y)]);
saveas(gcf,name_slice);
saveas(gcf,name_slice,'png');
close(fig_slice);
% Makes stream plot
fig_stream=figure;
streamline(x,y,current_x,current_y,startx1,starty);
axis([min(x),max(x),min(y),max(y)]);
saveas(gcf,name_stream);
saveas(gcf,name_stream,'png');
close(fig_stream);
end