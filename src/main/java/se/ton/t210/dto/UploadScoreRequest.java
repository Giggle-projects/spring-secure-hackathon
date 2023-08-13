package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.EvaluationItemScoreRecord;

import java.util.List;

@Getter
public class UploadScoreRequest {

    private Long evaluationItemId1;
    private int score1;

    private Long evaluationItemId2;
    private int score2;

    private Long evaluationItemId3;
    private int score3;

    private Long evaluationItemId4;
    private int score4;

    private Long evaluationItemId5;
    private int score5;

    public UploadScoreRequest(Long evaluationItemId1, int score1, Long evaluationItemId2, int score2, Long evaluationItemId3, int score3, Long evaluationItemId4, int score4, Long evaluationItemId5, int score5) {
        this.evaluationItemId1 = evaluationItemId1;
        this.score1 = score1;
        this.evaluationItemId2 = evaluationItemId2;
        this.score2 = score2;
        this.evaluationItemId3 = evaluationItemId3;
        this.score3 = score3;
        this.evaluationItemId4 = evaluationItemId4;
        this.score4 = score4;
        this.evaluationItemId5 = evaluationItemId5;
        this.score5 = score5;
    }

    public List<EvaluationItemScoreRecord> records(Long memberId) {
        return List.of(
            new EvaluationItemScoreRecord(memberId, evaluationItemId1, score1),
            new EvaluationItemScoreRecord(memberId, evaluationItemId2, score2),
            new EvaluationItemScoreRecord(memberId, evaluationItemId3, score3),
            new EvaluationItemScoreRecord(memberId, evaluationItemId4, score4),
            new EvaluationItemScoreRecord(memberId, evaluationItemId5, score5)
        );
    }

    public int scoreSum() {
        return score1 + score2 + score3 + score4 + score5;
    }
}
