clear

folder = 'E:\dataset\defense\defense-frames'; %file directory of the folder of folders of segmented videos per frame
list = dir(folder); %list of videos
n=length(list)-1; %number of videos
% for i=1:n  %uncomment for annotating all videos (1000) at once
for i = 199:202 %uncomment to choose videos to annotate
    
    infold = list(i).name(); %names of folders each video extracted to frames
    list2 = dir([folder '\' infold]); %list of frames (16 in total)
    if length(list2) == 12
        for j = 3:length(list2)
            filename = list2(j).name(1:end-4)
            I = imread([folder '\' infold '\' filename '.jpg']);
            imshow(I); title(filename);
            set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
            [X Y] = ginput(16); %choose 16 points in image
                %order: RFoot, RKnee, RHip, LHip, LK, LF, 
                        %Pelvis, Thorax, Neck, HeadTop, 
                        %RHand, RElbow, RShoulder, LS, LE, LH
%             V(:,:,j-2) = [X Y]; %uncomment for normal vids
%             V1(:,:,j-2)= [128-X Y]; %uncomment for flipped vids
            if j==4 %uncomment for baliw
                V(:,:,10) = [X Y];
            elseif j>4
                V(:,:,j-3) = [X Y];
            elseif j==3
                V(:,:,j-2) = [X Y];
            end
        end
    elseif length(list2) == 13
        for j = 3:length(list2)-1
            filename = list2(j).name(1:end-4)
            I = imread([folder '\' infold '\' filename '.jpg']);
            imshow(I); title(filename);
            set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
            [X Y] = ginput(16); %choose 16 points in image
                %order: RFoot, RKnee, RHip, LHip, LK, LF, 
                        %Pelvis, Thorax, Neck, HeadTop, 
                        %RHand, RElbow, RShoulder, LS, LE, LH
%             V(:,:,j-2) = [X Y]; %uncomment for normal vids
%             V1(:,:,j-2)= [128-X Y]; %uncomment for flipped vids
            if j==4 %uncomment for baliw 
                V(:,:,10) = [X Y];
            elseif j>4
                V(:,:,j-3) = [X Y];
            elseif j==3
                V(:,:,j-2) = [X Y];
            end
        end
    end
    %saves points into a json file (one per video)
    for k =1:16
        Z(:,:,k) = reshape(V(k,:,:),[2,10])';
%         Z1(:,:,k) = reshape(V1(k,:,:),[2,16])';
    end
    txt=jsonencode(table(Z(:,:,1),Z(:,:,2),Z(:,:,3),Z(:,:,4),Z(:,:,5),Z(:,:,6),Z(:,:,7),Z(:,:,8),Z(:,:,9),Z(:,:,10),Z(:,:,11),Z(:,:,12),Z(:,:,13),Z(:,:,14),Z(:,:,15),Z(:,:,16)));
%     txt1=jsonencode(table(Z1(:,:,6),Z1(:,:,5),Z1(:,:,4),Z1(:,:,3),Z1(:,:,2),Z1(:,:,1),Z1(:,:,7),Z1(:,:,8),Z1(:,:,9),Z1(:,:,10),Z1(:,:,16),Z1(:,:,15),Z1(:,:,14),Z1(:,:,13),Z1(:,:,12),Z1(:,:,11)));
    
    
    filenamefin = list2(4).name(1:end-7)
    fid = fopen(['E:\dataset\defense\defense\' filenamefin '.json'], 'w'); %input directory of folder with raw videos
    if fid == -1, error('Cannot create JSON file'); end
    fwrite(fid, txt, 'char');
    fclose(fid);
    
%     fid1 = fopen(['C:\Users\LJ\Desktop\dataset\run\run\' filenamefin '_flipped.json'], 'w'); %input directory of folder with raw videos
%     if fid1 == -1, error('Cannot create JSON file'); end
%     fwrite(fid1, txt1, 'char');
%     fclose(fid1);
    
end
