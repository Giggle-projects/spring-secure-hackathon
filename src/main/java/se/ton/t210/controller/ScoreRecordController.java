package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.MonthlyScoresResponse;
import se.ton.t210.dto.TopMonthlyScoresResponse;
import se.ton.t210.service.EvaluationScoreRecordService;

import java.time.LocalDate;

@RestController
public class ScoreRecordController {

    private final EvaluationScoreRecordService evaluationScoreRecordService;

    public ScoreRecordController(EvaluationScoreRecordService evaluationScoreRecordService) {
        this.evaluationScoreRecordService = evaluationScoreRecordService;
    }

    @GetMapping("/api/up")
    public String up() {
        return "299";
    }

    @GetMapping("/api/me/avgScores")
    public ResponseEntity<MonthlyScoresResponse> myScoreRecordResponse(String yearDate) {
        Long memberId = 1L;
        final MonthlyScoresResponse monthlyScoresResponse = evaluationScoreRecordService.averageScoresByJudgingItem(memberId, LocalDate.parse(yearDate));
        return ResponseEntity.ok(monthlyScoresResponse);
    }

    @GetMapping("/api/me/topScores")
    public ResponseEntity<TopMonthlyScoresResponse> topScoresRecordResponse(ApplicationType applicationType, String yearDate) {
        final TopMonthlyScoresResponse topMonthlyScoresResponse = evaluationScoreRecordService.averageAllScoresByJudgingItem(applicationType, LocalDate.parse(yearDate));
        return ResponseEntity.ok(topMonthlyScoresResponse);
    }
}
