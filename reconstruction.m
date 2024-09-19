function simplifiedPoints = reconstruction(points, epsilon)
    % douglasPeucker多边形拟合算法
    % 输入参数：
    % points: 输入点集，每行一个点 [x, y]
    % epsilon: 控制简化程度的参数，越小则曲线越精确，但点数越多
    % 输出参数：
    % simplifiedPoints: 简化后的点集

    % 如果点集中的点数小于3，无法进行进一步简化
    if size(points, 1) < 3
        simplifiedPoints = points;
        return;
    end

    % 查找曲线中距离起点和终点最远的点
    dMax = 0;
    index = 0;
    for i = 2:(size(points, 1) - 1)
        d = pointToLineDistance(points(i, :), points(1, :), points(end, :));
        if d > dMax
            dMax = d;
            index = i;
        end
    end

    % 如果最远的点距离大于阈值epsilon，递归地简化左右两段曲线
    if dMax > epsilon
        leftPoints = points(1:index, :);
        rightPoints = points(index:end, :);
        simplifiedLeft = douglasPeucker(leftPoints, epsilon);
        simplifiedRight = douglasPeucker(rightPoints, epsilon);
        simplifiedPoints = [simplifiedLeft(1:end-1, :); simplifiedRight];
    else
        % 如果距离都小于阈值epsilon，则直接连接起点和终点
        simplifiedPoints = [points(1, :); points(end, :)];
    end
end

function distance = pointToLineDistance(point, lineStart, lineEnd)
    % 计算点到线段的距离
    a = lineStart - lineEnd;
    b = point - lineEnd;
    distance = norm(cross([a, 0], [b, 0])) / norm(a);
end
