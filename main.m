clc;
clear;

datasets = {'jaffe.csv'};

for d=1:length(datasets)
   
    dataset_name = datasets{d};
    path = strcat(dataset_name);
    dataset = load(path);
    [~, fNum] = size(dataset);
    fNum = fNum - 1;
    
    fRange = 10:10:100;
    bucketNum = length(fRange);

    iters = 20;
    ESACO_accuracy = zeros(iters,10);
    ESACO_Precision = zeros(iters,10);
    ESACO_F1_score = zeros(iters,10);
    ESACO_Specificity = zeros(iters,10);
    ESACO_Sensitivity = zeros(iters,10);
    ESACO_FalsePositiveRate = zeros(iters,10);
    time_ESACO = zeros(1,iters);
 
    for i=1:iters
        disp(i);
        [dataTrain, dataTrainLabel, dataTest, dataTestLabel] = SplitDataset(dataset);
        
        % randomly miss n% of class labels and replace them with NaN!
        labeledAndUnlabeled = MissDatasetLabel1(dataTrainLabel, 0.2);
        
        labeled = find(~isnan(labeledAndUnlabeled));
        labeledData = [dataTrain(labeled, :), labeledAndUnlabeled(labeled)];
        [~,columns]=size(labeledData);
        unlabeled = find(isnan(labeledAndUnlabeled));
        unlabeledData = dataTrain(unlabeled, :);       

%% Ant-td
        start=tic;       
        % calculate features correlation
        fCorr = abs( 1 - pdist2(dataTrain', dataTrain', 'cosine') );
        fCorr(fCorr == 0) = 0.0001;
        f1=max(fCorr,[],2);
        %flCorr = abs( 1 - pdist2(labeledData(:,1:end-1)', labeledData(:,end)', 'cosine') );
        X=labeledData(:,1:end-1);
        [~,feature_num]=size(X);  
        f2=zeros(feature_num,1);
        Y=labeledData(:,end);
        for q=1:feature_num
            %data=labeledData(q,1:end-1);
            answ=mine(X(:,q)',Y');
            f2(q,1)=answ.mic;
        end 
        f3=zeros(feature_num,1);
        Y=labeledData(:,end);
        for q=1:feature_num
            %data=labeledData(q,1:end-1);
            f3(q,1)=SU(X(:,q),Y);
        end 
 
      
        %Measure 13:llcfs
         % Feature Selection and Kernel Learning for Local Learning-Based Clustering
        M13 = llcfs( dataTrain );
        [~, R13] = sort(M13, 'descend');
        [~,R5] = sort(f1,'ascend');
       [~,R18] = sort(f2,'descend');
      [~,R1] = sort(f2,'descend');

        
        %% MCDM Process

        P1=[R5,R13,R1,R18];
        [m,n]=size(P1);
        w=ones(1,n);
        W = w./sum(w);
        RH5=zeros(1,m);
        b2=R5;
        for q=1:m
            n10=b2(q);
            RH5(n10)=q;
        end
        
        RH1=zeros(1,m);
        b4=R1;
        for q=1:m
            n10=b4(q);
            RH1(n10)=q;
        end
        
        RH13=zeros(1,m);
        b4=R13;
        for q=1:m
            n10=b4(q);
            RH13(n10)=q;
        end
        
        RH18=zeros(1,m);
        b3=R18;
        for q=1:m
            n10=b3(q);
            RH18(n10)=q;
        end
        
        P8=[RH18',RH5',RH13',RH1'];
       % E4=Top(P8,ones(1,feature_num),W);
       MM=ones(1,4);
       [E4]=Fun_MOORA(m,n,P8,W,MM,4 );

        % Ant Colony Process
        
        pheromone = ESACO(dataTrain, E4', []);
        [ph_val1, S9] = sort(-pheromone);
         time_ESACO(d,i)=toc(start);
%% Training Phase        
        for j=1:bucketNum
            disp(j);               
             % Ant
            ESACO_Result(i, j) = Classification('knn', dataTrain(:, S9(1:fRange(j))), ...
            dataTrainLabel, dataTest(:, S9(1:fRange(j))), dataTestLabel);
            ESACO_accuracy(i,j)=ESACO_Result(i,j).Accuracy;
            ESACO_Precision(i,j)=ESACO_Result(i,j).Precision;
            ESACO_F1_score(i,j)=ESACO_Result(i,j).F1_score;
            ESACO_Specificity(i,j)=ESACO_Result(i,j).Specificity;
            ESACO_Sensitivity(i,j)=ESACO_Result(i,j).Sensitivity;
            ESACO_FalsePositiveRate(i,j)=ESACO_Result(i,j).FalsePositiveRate;              
        end

    end

    
end
