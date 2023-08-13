package se.ton.t210.dto;

import lombok.Getter;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Getter
public class MonthlyScoresResponse {

    private final List<AverageScoresByEvaluationItem> avgScoresByEvaluationItems;

    public MonthlyScoresResponse(List<AverageScoresByEvaluationItem> avgScoresByEvaluationItems) {
        this.avgScoresByEvaluationItems = avgScoresByEvaluationItems;
    }

    public static MonthlyScoresResponse listOf(Map<Long, Double> averageScoresByEvaluationItem) {
        return new MonthlyScoresResponse(averageScoresByEvaluationItem.keySet().stream()
            .map(it -> new AverageScoresByEvaluationItem(it, averageScoresByEvaluationItem.get(it)))
            .collect(Collectors.toList())
        );
    }
}
