package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.MonthlyScore;

import java.util.List;

@Getter
public class MyScoreResponse {

    private final int score;
    private final int maxScore;

    public MyScoreResponse(int score, int maxScore) {
        this.score = score;
        this.maxScore = maxScore;
    }

    public static MyScoreResponse of(List<MonthlyScore> scores) {
        final int lastScore = scores.get(scores.size() - 1).getScore();
        final int maxScore = scores.stream()
            .mapToInt(MonthlyScore::getScore)
            .max()
            .orElse(0);
        return new MyScoreResponse(lastScore, maxScore);
    }
}
