package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.domain.type.Gender;
import se.ton.t210.dto.*;
import se.ton.t210.service.EvaluationScoreRecordService;

import java.time.LocalDate;
import java.util.List;

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

    @GetMapping("/api/me/scores")
    public ResponseEntity<ScoreResponse> myScores() {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
        final ScoreResponse scoreResponse = evaluationScoreRecordService.myScore(member.getId(), LocalDate.now());
        return ResponseEntity.ok(scoreResponse);
    }

    @PostMapping("/api/me/scores")
    public ResponseEntity<ScoreResponse> uploadScore(@RequestBody UploadScoreRequest request) {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
        final ScoreResponse scoreResponse = evaluationScoreRecordService.uploadScore(request, member.getId());
        return ResponseEntity.ok(scoreResponse);
    }

    @GetMapping("/api/application/rank")
    public ResponseEntity<List<RankResponse>> rank() {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
        final List<RankResponse> rankResponses = evaluationScoreRecordService.rank(member.getApplicationType(), 5);
        return ResponseEntity.ok(rankResponses);
    }

    @GetMapping("/api/records/count")
    public ResponseEntity<RecordCountResponse> recordCount() {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
        final RecordCountResponse countResponse = evaluationScoreRecordService.count(member.getApplicationType());
        return ResponseEntity.ok(countResponse);
    }
}
