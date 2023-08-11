package se.ton.t210.dto;

import lombok.Getter;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Getter
public class MonthlyScoresResponse {

    private final List<AverageScoresByJudgingItem> avgScoresByJudgingItems;

    public MonthlyScoresResponse(List<AverageScoresByJudgingItem> avgScoresByJudgingItems) {
        this.avgScoresByJudgingItems = avgScoresByJudgingItems;
    }

    public static MonthlyScoresResponse listOf(Map<Long, Double> averageScoresByJudgingItem) {
        return new MonthlyScoresResponse(averageScoresByJudgingItem.keySet().stream()
            .map(it -> new AverageScoresByJudgingItem(it, averageScoresByJudgingItem.get(it)))
            .collect(Collectors.toList())
        );
    }
}
