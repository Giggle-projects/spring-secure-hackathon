package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.MonthlyScore;

import java.util.List;

@Getter
public class MyScoreResponse {

    private final float score;
    private final float maxScore;

    public MyScoreResponse(float score, float maxScore) {
        this.score = score;
        this.maxScore = maxScore;
    }

    public static MyScoreResponse of(List<MonthlyScore> scores) {
        if (scores.isEmpty()) {
            return new MyScoreResponse(0, 0);
        }
        final int lastScore = scores.get(scores.size() - 1).getScore();
        final int maxScore = scores.stream()
                .mapToInt(MonthlyScore::getScore)
                .max()
                .orElse(0);
        return new MyScoreResponse(lastScore, maxScore);
    }
}
