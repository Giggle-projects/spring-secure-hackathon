package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.domain.type.Gender;
import se.ton.t210.dto.*;
import se.ton.t210.service.ScoreService;

import java.time.LocalDate;
import java.util.List;

@RestController
public class ScoreController {

    private final ScoreService scoreService;

    public ScoreController(ScoreService scoreService) {
        this.scoreService = scoreService;
    }

    @GetMapping("/api/score/count")
    public ResponseEntity<RecordCountResponse> recordCount() {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
        final RecordCountResponse countResponse = scoreService.count(member.getApplicationType());
        return ResponseEntity.ok(countResponse);
    }

    @GetMapping("/api/score")
    public ResponseEntity<ScoreResponse> evaluationScore(Long evaluationItemId, Integer score) {
        final int evaluationScore = scoreService.evaluationScore(evaluationItemId, score);
        return ResponseEntity.ok(new ScoreResponse(evaluationScore));
    }

    @GetMapping("/api/score/me")
    public ResponseEntity<ScoreResponse> myScore() {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
        final ScoreResponse scoreResponse = scoreService.score(member.getId(), LocalDate.now());
        return ResponseEntity.ok(scoreResponse);
    }

    @PostMapping("/api/score/me")
    public ResponseEntity<ScoreResponse> updateScore(@RequestBody List<EvaluationScoreRequest> request) {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
        final ScoreResponse scoreResponse = scoreService.update(member.getId(), request, LocalDate.now());
        return ResponseEntity.ok(scoreResponse);
    }

    @GetMapping("/api/score/rank")
    public ResponseEntity<List<RankResponse>> rank(Integer rankCnt) {
        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
        final List<RankResponse> rankResponses = scoreService.rank(member.getApplicationType(), rankCnt, LocalDate.now());
        return ResponseEntity.ok(rankResponses);
    }
}
