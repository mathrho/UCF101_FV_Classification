function darwin = videodarwin(feat, opt)
    
    if size(feat,2)==1 & sum(feat,1)==0
        darwin = zeros(size(feat,1), 1);
        
        if opt.forward_reverse
            darwin = [darwin; darwin];
        end
        return;
    end

    % FORWARD
    feat_forw = getFeatVector(feat, opt.feat_type);
    feat_forw = getNonLinearity(feat_forw, opt.nonlinear);
    darwin_forw = liblinearsvr(feat_forw, opt.cval, opt.normalizer);
    darwin = darwin_forw;

    % viol_constraints = ... ;
    % active_constraints = ... ;
    % darwin_forw_segm = darwin_forw - eta * (fisher(:, viol_constraints & active_constraints));

    % BACKWARD
    if opt.forward_reverse
        % feat_rev = flip(feat, 2);
        order = 1:size(feat,2);
        [~,order] = sort(order,'descend');
        feat_rev = feat(:,order);

        feat_rev = getFeatVector(feat_rev, opt.feat_type);
        feat_rev = getNonLinearity(feat_rev, opt.nonlinear);
        darwin_rev = liblinearsvr(feat_rev, opt.cval, opt.normalizer);
        darwin = [darwin_forw; darwin_rev];
    end

end


function w = liblinearsvr(data,C,normalizer)

    % normalization by column
    if strcmp(normalizer, 'l2')
        data = normalizeL2(data);
    else
        fprintf('Do not support (%s) Normalization yet\n', normalizer);
    end

    N = size(data,2);
    labels = [1:N]';
    model = train(double(labels), sparse(double(data')), sprintf('-c %1.6f -s 11 -q',C));
    w = model.w';
    
end


function data = getFeatVector(data,feat_type)

    if strcmp(feat_type, 'tv-mean')
        data = cumsum(data,2);
%       feat_rev = feat_rev ./ repmat(1:size(feat_rev, 2), size(feat_rev,
%       1), 1) ; % NOT REALLY NEEDED, CANCELLED IN THE L2 NORMALIZATION
        %data = data(:,2:end); % why have to remove the first frame?
    else
        fprintf('Do not support (%s) feature vector yet\n', feat_type);
    end
    %if strcmp(data, 'ind')
        % DO NOTHING
    %end
    %if strcmp(data, 'movavg')
        % TODO
    %end
    %if strcmp(data, 'tv-max')
        % TODO
    %end

end


function data = getNonLinearity(data,nonlinear)
    
    if strcmp(nonlinear, 'possqrt')
        data(data<0) = 0;
        data = sqrt(data);
    elseif strcmp(nonlinear, 'sqrt')
        data = sqrt(data);
    elseif strcmp(nonlinear, 'abssqrt')
        data = sqrt(abs(data));
    elseif strcmp(nonlinear, 'signsqrt')
        data = sign(data).*sqrt(abs(data));
    end

end


function x = normalizeL2(x)
    x=x./repmat(sqrt(sum(x.*conj(x),1)),size(x,1),1);
end

