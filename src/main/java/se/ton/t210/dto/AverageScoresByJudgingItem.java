package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class AverageScoresByJudgingItem {

    private final Long judgingId;
    private final double avgScore;

    public AverageScoresByJudgingItem(Long judgingId, double avgScore) {
        this.judgingId = judgingId;
        this.avgScore = avgScore;
    }
}
