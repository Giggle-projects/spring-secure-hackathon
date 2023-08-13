package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class UploadScoreRequest {

    private Long evaluationItemId1;
    private Long score1;

    private Long evaluationItemId2;
    private Long score2;

    private Long evaluationItemId3;
    private Long score3;

    private Long evaluationItemId4;
    private Long score4;

    private Long evaluationItemId5;
    private Long score5;

    public UploadScoreRequest(Long evaluationItemId1, Long score1, Long evaluationItemId2, Long score2, Long evaluationItemId3, Long score3, Long evaluationItemId4, Long score4, Long evaluationItemId5, Long score5) {
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

    public Long scoreSum() {
        return score1 + score2 + score3 + score4 + score5;
    }
}
