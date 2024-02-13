function loca_improved_kmeans(I)
%————————pre-processing————————
th_min = min(min(I));
th_max = max(max(I));
I = imadjust(I,[th_min,th_max],[0,1]);
I= medfilt2(I,[3 3],'symmetric');
%__________________________________________________________________________
[m, n, p] = size(I);
%——————————feature space————————————————
wavelength = 2.^(0:1) * 3;
orientation = 0:45:135;
g = gabor(wavelength,orientation);
IS = im2single(I);
gabormag = imgaborfilt(IS,g);
% figure()
% montage(gabormag,'Size',[2 4])
for i = 1:length(g)
    sigma = 0.5*g(i).Wavelength;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),3*sigma); 
end
%  figure()
%  montage(gabormag,'Size',[2 4])
texture =  im2double(gabormag(:,:,i));
nrows = size(I,1);
ncols = size(I,2);
[X,Y] = meshgrid(1:ncols,1:nrows);
featureSet1 = cat(3,I,gabormag);
featureSet2 = cat(3,featureSet1,X);
featureSet = cat(3,featureSet2,Y);
%  featureSet = cat(3,I,texture);
X = im2double(featureSet);
%——————————————————————————————————
X = reshape(double(I), m*n, p);
rng('default');
%------------cluster centre---------
    T=0.5*(min(I(:))+max(I(:)));
    done=false;
    while(~done)
        g=I>=T;
        Tnext = 0.5*(mean(I(g))+mean(I(~g)));
        done=abs(T-Tnext)<0.5;
        T=Tnext;
    end
    g=I>=T;
    C =[mean(I(g));mean(I(~g))];

k = 2;
[C, label, J2] = kmeans(I, k);
% label1 = reshape(label,m,n,p);
% RGB1 = label2rgb(label1); 
% figure
% imshow(RGB1)
I_seg2 = reshape(C(label, :), m, n, p);

% figure
% subplot(1, 3, 1), imshow(I);
% subplot(142);
% imshow(I_seg2)
%--------------------------post-processing----------------------------------
% thresh = (min(min(I_seg2))+max(max(I_seg2)))/2;
thresh = graythresh(I_seg2);
f=im2bw(I_seg2,thresh);
% figure(),
% imshow(f),

se = strel('disk',3);
fd=imdilate(f,se);
% figure();
% imshow(fd);
fo=imerode(fd,se);
% figure(),
% imshow(fo),
L = bwlabel(~fo);  
RGB2 = label2rgb(L); 

% subplot(1,4,3)
% imshow(RGB2)
% title('connected regions')
stats = regionprops(L, 'Area','Centroid','BoundingBox');  
[~,index] = sort([stats.Area], 'descend'); 
% subplot(8,6,5)
% imshow(~fo)
% title('labeled connected regions')
hold on
numObj = numel(stats);
for k = 1 : numObj

      text(stats(k).Centroid(1),stats(k).Centroid(2), ...
      int2str(k),'Color','b');
    rectangle('Position',stats(k).BoundingBox,'EdgeColor','r');
end
hold off
n1 = 1; 
if length(stats) > 2
   I2 = ismember(L, index(1:n1)); 
else
   I2 = L; 
end

se = strel('disk',7);
fo1=imerode(I2,se);
fd1=imdilate(fo1,se);
fd2=imdilate(fd1,se);
fd3=imdilate(fd2,se);
fd4=imdilate(fd3,se);
fo2=imerode(fd4,se);
fo3=imerode(fo2,se);
I2=imerode(fo3,se);
I_edge= bwperim(I2,8);

    R=rgb(:,:,1);
    G=rgb(:,:,2);
    B=rgb(:,:,3);
    re(:,:,1)=R.*double(I2);
    re(:,:,2)=G.*double(I2);
    re(:,:,3)=B.*double(I2);
figure()
imshow(re)
% figure()
% imshow(I_edge)
w = uint8(1)
row = 0;
col = 0;
figure()
for r1 =1:m
    for c1 = 1:n
      if uint8(I_edge(r1,c1)) == 1
        row(w) = r1;
        col(w) = c1;
        w = w+1;
      end
    end
end
nn = length(row);
imshow(rgb)
s = regionprops(I2,'centroid','MajorAxisLength','MinorAxisLength');
centroids = cat(1,s.Centroid);
hold on
plot(centroids(1),centroids(2),'go','MarkerSize',3,'LineWidth',2)
hold on
for kk =1:nn

% plot(centroids(1),centroids(2),col(1),row(1),'^g');
% line([centroids(1),col(1)],[centroids(2),row(1)]);
% a = abs((row(1)-centroids(2))/(col(1)-centroids(1)));
% b = abs(a.*centroids(1)-centroids(2));
p = polyfit([centroids(1),col(1)],[centroids(2),row(1)],1);
x = [centroids(1),col(1)];
y = round(polyval(p,x));
hold on
line(x,y,'Color','red','LineWidth',2);
text((centroids(1)+col(1))/2+8,(centroids(2)+row(1))/2,'R','Color','blue','FontSize',14);
end 
%  x_inter = intersect(x,col);
% 
% hold on
% plot(229,351,'oy')

% 
% 
diameters = mean([s.MajorAxisLength s.MinorAxisLength],2);
hold on
viscircles(centroids,diameters/2);
hold off
    figure()
    subplot(144)
    imshow(re);
     title('defect precise location');
% calculating area img
defectimg(num) = (25/ (400*400)).*sum(sum(im2double(I2)));
