package se.ton.t210.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import se.ton.t210.dto.MemberAverageScoresByJudgingItemResponse;
import se.ton.t210.service.ScoreRecordService;

import java.time.Month;

@RestController
public class ScoreRecordController {

    private final ScoreRecordService scoreRecordService;

    public ScoreRecordController(ScoreRecordService scoreRecordService) {
        this.scoreRecordService = scoreRecordService;
    }

    @GetMapping("/api/me/avgScores")
    public MemberAverageScoresByJudgingItemResponse myScoreRecordResponse(Month month) {
        Long memberId = 1L;
        return scoreRecordService.averageScoresByJudgingItem(memberId, month);
    }
}
