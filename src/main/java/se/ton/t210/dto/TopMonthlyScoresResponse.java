package se.ton.t210.dto;

import lombok.Getter;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Getter
public class TopMonthlyScoresResponse {

    private final MonthlyScoresResponse top50ScoresByJudgingItems;
    private final MonthlyScoresResponse top30ScoresByJudgingItems;

    public TopMonthlyScoresResponse(MonthlyScoresResponse top50ScoresByJudgingItems,
                                    MonthlyScoresResponse top30ScoresByJudgingItems) {
        this.top50ScoresByJudgingItems = top50ScoresByJudgingItems;
        this.top30ScoresByJudgingItems = top30ScoresByJudgingItems;
    }

    public static TopMonthlyScoresResponse listOf(Map<Long, Double> top50ScoresByJudgingItemMap, Map<Long, Double> top30ScoresByJudgingItemMap) {
        final MonthlyScoresResponse monthlyTop50ScoresResponse = MonthlyScoresResponse.listOf(top50ScoresByJudgingItemMap);
        final MonthlyScoresResponse monthlyTop30ScoresResponse = MonthlyScoresResponse.listOf(top30ScoresByJudgingItemMap);
        return new TopMonthlyScoresResponse(monthlyTop50ScoresResponse, monthlyTop30ScoresResponse);
    }
}
