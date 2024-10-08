function simplifiedPoints = reconstruction(points, epsilon)
    % Douglas Peucker polygon fitting algorithm
    % Input parameters:
    % points: Input point set, one point per line [x, y]
    % epsilon: Parameter that controls the degree of simplification, the smaller the epsilon, the more accurate the curve, but the more points
    % Output parameters:
    % simplifiedPoints: Simplified point set

    if size(points, 1) < 3
        simplifiedPoints = points;
        return;
    end

    dMax = 0;
    index = 0;
    for i = 2:(size(points, 1) - 1)
        d = pointToLineDistance(points(i, :), points(1, :), points(end, :));
        if d > dMax
            dMax = d;
            index = i;
        end
    end

    if dMax > epsilon
        leftPoints = points(1:index, :);
        rightPoints = points(index:end, :);
        simplifiedLeft = douglasPeucker(leftPoints, epsilon);
        simplifiedRight = douglasPeucker(rightPoints, epsilon);
        simplifiedPoints = [simplifiedLeft(1:end-1, :); simplifiedRight];
    else
        simplifiedPoints = [points(1, :); points(end, :)];
    end
end

function distance = pointToLineDistance(point, lineStart, lineEnd)
    % Calculate the distance from a point to a line segment
    a = lineStart - lineEnd;
    b = point - lineEnd;
    distance = norm(cross([a, 0], [b, 0])) / norm(a);
end
