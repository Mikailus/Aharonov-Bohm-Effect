function make_density_plot = draw(path_x,path_y,path_density,name_pcolor)
%Creates pcolor plot for density

%Gets data from files
[x,delimiterOut]=importdata(path_x);
[y,delimiterOut]=importdata(path_y);
[z,delimiterOut]=importdata(path_density);

% Makes pcolor plot
fig_gest=figure;
pcolor(x,y,z);
shading flat;
colorbar
xlabel('x');
ylabel('y');
%axis([min(x),max(x),min(y),max(y)]);
saveas(gcf,name_pcolor);
saveas(gcf,name_pcolor,'png');
close(fig_gest);
end
