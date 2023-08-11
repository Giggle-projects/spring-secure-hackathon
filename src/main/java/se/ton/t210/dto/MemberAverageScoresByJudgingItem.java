package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class MemberAverageScoresByJudgingItem {

    private final Long judgingId;
    private final double avgScore;

    public MemberAverageScoresByJudgingItem(Long judgingId, double avgScore) {
        this.judgingId = judgingId;
        this.avgScore = avgScore;
    }
}
