package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.MonthlyScore;

@Getter
public class MonthlyScoreResponse {

    private final int month;
    private final int score;

    public MonthlyScoreResponse(int month, int score) {
        this.month = month;
        this.score = score;
    }

    public static MonthlyScoreResponse of(MonthlyScore score) {
        return new MonthlyScoreResponse(
            score.getYearMonth().getMonth().getValue(),
            score.getScore()
        );
    }
}
