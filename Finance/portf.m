function portf(portfolioType)
    % Загрузка данных
    df = readtable('2.xlsx', 'VariableNamingRule', 'preserve');
    assetSymbols = {'GAZP', 'FEES', 'LKOH', 'SBER03', 'ROSN', 'VTBR'};
    priceData = df{1:23, assetSymbols};

    % Ограничение суммы весов портфеля (сумма = 1)
    sumWeightsConstraint = ones(1, 6);
    sumWeightsValue = 1;

    % Расчет доходностей
    numAssets = length(assetSymbols);
    numPeriods = 22;
    annualizedReturns = zeros(numAssets, numPeriods);
    meanReturns = [];
    for assetIdx = 1:numAssets
        prices = priceData(:, assetIdx);
        for periodIdx = 2:23
            annualizedReturns(assetIdx, periodIdx-1) = ((prices(periodIdx) - prices(1)) / prices(1)) * 365 / (periodIdx - 1);
        end
        meanReturns = [meanReturns, mean(annualizedReturns(assetIdx, :))];
    end

    % Ковариационная матрица доходностей
    covarianceMatrix = cov(annualizedReturns');

    % ---------------------------------------------------------------------
    % ОГРАНИЧЕНИЯ НА ПОРТФЕЛЬ
    % ---------------------------------------------------------------------
    if portfolioType == 1
        % ================================================================
        % Тип 1: Индивидуальные двусторонние ограничения на каждый актив
        % ================================================================
        % 0.05 <= x_i <= 0.39 для каждого актива

        lowerBounds = 0.05 * ones(1, numAssets);
        upperBounds = 0.39 * ones(1, numAssets);

        inequalityMatrix = [-eye(numAssets); eye(numAssets)];
        inequalityVector = [-lowerBounds, upperBounds]';

    elseif portfolioType == 2
        % ================================================================
        % Тип 2: Групповые ограничения
        % ================================================================
        % Порядок групп:
        % 1. Нефте-газовый сектор (GAZP, LKOH, ROSN)
        % 2. Энергетика (FEES)
        % 3. Банки (SBER03, VTBR)
        % 4. Внутренний рынок (LKOH, FEES, SBER03, VTBR)
        % 5. Внешний рынок (GAZP, ROSN)

        % Групповые нижние и верхние границы
        groupLowerBounds = [0.25, 0.27, 0.15, 0.25, 0.10];
        groupUpperBounds = [0.65, 0.75, 0.55, 0.85, 0.35];

        % Матрица принадлежности активов к группам
        groupMatrix = [
            1, 0, 1, 0, 1, 0;  % 1. Нефте-газовый сектор
            0, 1, 0, 0, 0, 0;  % 2. Энергетика
            0, 0, 0, 1, 0, 1;  % 3. Банки
            0, 1, 1, 1, 0, 1;  % 4. Внутренний рынок
            1, 0, 0, 0, 1, 0   % 5. Внешний рынок
        ];

        % Групповые ограничения: groupLowerBounds <= groupMatrix*x <= groupUpperBounds
        % Индивидуальные ограничения: 0 <= x_i <= 1
        inequalityMatrix = [
            -groupMatrix;       % -groupMatrix*x <= -groupLowerBounds
             groupMatrix;       %  groupMatrix*x <= groupUpperBounds
            -eye(numAssets);    % -x_i <= 0
             eye(numAssets)     %  x_i <= 1
        ];
        inequalityVector = [
            -groupLowerBounds'; % -groupLowerBounds
             groupUpperBounds'; %  groupUpperBounds
             zeros(numAssets,1);% 0
             ones(numAssets,1)  % 1
        ];

    elseif portfolioType == 3
        % ================================================================
        % Тип 3: Ограничения на отношения между группами
        % ================================================================
        % Формируем две группы для сравнения:
        % Нефте-газровый сектор (G1): GAZP, LKOH, ROSN
        % Энергетика и банки (G2): FEES, SBER03, VTBR
        % Ограничения: 0.2 <= sum(x_A) / sum(x_B) <= 0.75

        % Матрицы для сравнения групп
        % G1: группы для числителя, G2: группы для знаменателя
        G1 = [
            1, 0, 1, 0, 1, 0;  % Нефте-газовый сектор
            0, 1, 0, 0, 0, 0;  % Энергетика
            0, 0, 0, 1, 0, 1;  % Банки
            0, 1, 1, 1, 0, 1;  % Внутренний рынок
            1, 0, 0, 0, 1, 0   % Внешний рынок
        ];

        G2 = [
            0, 1, 0, 1, 0, 1;  % FEES, SBER03, VTBR
            1, 0, 1, 0, 1, 0;  % GAZP, LKOH, ROSN
            0, 1, 0, 1, 0, 1;  % FEES, SBER03, VTBR
            1, 0, 1, 0, 1, 0;  % GAZP, LKOH, ROSN
            0, 1, 0, 1, 0, 1   % FEES, SBER03, VTBR
        ];

        % Вектор нижних и верхних границ для отношения групп
        groupRatioLower = [0.2, 0.2, 0.2, 0.2, 0.2];
        groupRatioUpper = [0.75, 0.75, 0.75, 0.75, 0.75];

        % Формируем ограничения:
        % (G1*x) - lower * (G2*x) <= 0
        % -(G1*x) + upper * (G2*x) <= 0
        groupComparisonMatrix = [];
        groupComparisonVector = [];
        for i = 1:5
            % (G1_i*x) - lower_i*(G2_i*x) <= 0
            groupComparisonMatrix = [groupComparisonMatrix; G1(i,:) - groupRatioLower(i)*G2(i,:)];
            groupComparisonVector = [groupComparisonVector; 0];
            % -(G1_i*x) + upper_i*(G2_i*x) <= 0
            groupComparisonMatrix = [groupComparisonMatrix; -G1(i,:) + groupRatioUpper(i)*G2(i,:)];
            groupComparisonVector = [groupComparisonVector; 0];
        end

        % Индивидуальные ограничения: 0 <= x_i <= 1
        individualMatrix = [-eye(numAssets); eye(numAssets)];
        individualVector = [zeros(numAssets,1); ones(numAssets,1)];

        % Собираем все ограничения
        inequalityMatrix = [groupComparisonMatrix; individualMatrix];
        inequalityVector = [groupComparisonVector; individualVector];

    else
        error('Неверный тип ограничений');
    end

    % Оптимизация портфеля при различных коэффициентах неприятия риска
    numThetaSteps = 7;
    portfolioReturns = zeros(1, numThetaSteps);
    stepCounter = 0;

    for thetaValue = 20:5:55
        stepCounter = stepCounter + 1;
        options = optimset('Display','off');

        % Матрица риска
        riskMatrix = (thetaValue / 10) * covarianceMatrix;
        linearTerm = -meanReturns';

        % Решение задачи квадратичного программирования
        optimalWeights = quadprog(riskMatrix, linearTerm, inequalityMatrix, inequalityVector, sumWeightsConstraint, sumWeightsValue, [], [], [], options);

        % Расчет доходности портфеля
        portfolioReturnPercent = round(dot(meanReturns, optimalWeights) * 100, 3);

        assetNames = df.Properties.VariableNames(4:9);

        % Вывод результатов
        fprintf('Theta = %.2f | Доходность = %.3f%%\n', thetaValue / 10, portfolioReturnPercent);
        fprintf('Распределение по компаниям:\n');
        for idx = 1:length(assetNames)
            fprintf('  %s: %.4f\n', assetNames{idx}, optimalWeights(idx));
        end
        fprintf('-------------------------------------\n');

        portfolioReturns(stepCounter) = portfolioReturnPercent;
    end

    % График зависимости доходности от коэффициента неприятия риска
    figure;
    thetaAxis = 20:5:55;
    plot(thetaAxis / 10, portfolioReturns, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    grid on;
    xlabel('Коэффициент неприятия риска', 'FontSize', 14, 'Interpreter', 'tex');
    ylabel('Доходность', 'FontSize', 14, 'Interpreter', 'tex');
    set(gca, 'FontSize', 12, 'GridAlpha', 0.3);
    
    y_shift = 0.02 * range(portfolioReturns);
    for i = 1:length(thetaAxis)
        text(thetaAxis(i)/10, portfolioReturns(i) + y_shift, ...
            sprintf('%.2f%%', portfolioReturns(i)), ...
            'VerticalAlignment', 'bottom', ...
            'HorizontalAlignment', 'center', ...
            'FontSize', 10);
    end
    
    ax = gca;
    ax.Position(2) = ax.Position(2) + 0.05;
    ax.Position(4) = ax.Position(4) - 0.05;
    
    filename = sprintf('plot_%d.png', portfolioType);
    saveas(gcf, filename);
end