package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.domain.type.Gender;
import se.ton.t210.dto.MonthlyScoresResponse;
import se.ton.t210.dto.RecordCountResponse;
import se.ton.t210.dto.ScoreResponse;
import se.ton.t210.dto.TopMonthlyScoresResponse;
import se.ton.t210.dto.UploadScoreRequest;
import se.ton.t210.service.EvaluationScoreRecordService;

import java.time.LocalDate;

@RestController
public class EvaluationScoreRecordController {

    private final EvaluationScoreRecordService evaluationScoreRecordService;

    public EvaluationScoreRecordController(EvaluationScoreRecordService evaluationScoreRecordService) {
        this.evaluationScoreRecordService = evaluationScoreRecordService;
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

    @GetMapping("/api/records/count")
    public ResponseEntity<RecordCountResponse> recordCount() {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
        final RecordCountResponse countResponse = evaluationScoreRecordService.count(member.getApplicationType());
        return ResponseEntity.ok(countResponse);
    }

    @GetMapping("/api/me/scores")
    public ResponseEntity<ScoreResponse> myScores() {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
        final ScoreResponse scoreResponse = evaluationScoreRecordService.myScore(member.getId(), LocalDate.now());
        return ResponseEntity.ok(scoreResponse);
    }

    @PostMapping("/api/me/scores")
    public ResponseEntity<Void> uploadScore(@RequestBody UploadScoreRequest request) {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
        final ScoreResponse scoreResponse = evaluationScoreRecordService.uploadScore(request, member.getId(), LocalDate.now());
        return ResponseEntity.ok(scoreResponse);
    }
}
