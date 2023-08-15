package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.dto.*;
import se.ton.t210.service.ScoreService;

import javax.validation.Valid;
import javax.validation.constraints.NotNull;
import java.time.LocalDate;
import java.util.List;

@Validated
@RestController
public class ScoreController {

    private final ScoreService scoreService;

    public ScoreController(ScoreService scoreService) {
        this.scoreService = scoreService;
    }

    @GetMapping("/api/score/count")
    public ResponseEntity<ScoreCountResponse> scoreCount() {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.PoliceOfficerMale);
        final ScoreCountResponse countResponse = scoreService.count(member.getApplicationType());
        return ResponseEntity.ok(countResponse);
    }

    @GetMapping("/api/score/evaluate")
    public ResponseEntity<ScoreResponse> evaluationScore(@Valid @NotNull Long evaluationItemId,
                                                         @Valid @NotNull Integer score) {
        final int evaluationScore = scoreService.evaluate(evaluationItemId, score);
        return ResponseEntity.ok(new ScoreResponse(evaluationScore));
    }

    @GetMapping("/api/score/year")
    public ResponseEntity<List<MonthlyScoreResponse>> scores() {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.PoliceOfficerMale);
        final List<MonthlyScoreResponse> scoresYear = scoreService.scoresYear(member, LocalDate.now());
        return ResponseEntity.ok(scoresYear);
    }

    @GetMapping("/api/score/expect")
    public ResponseEntity<ExpectScoreResponse> expect() {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.PoliceOfficerMale);
        final ExpectScoreResponse scoreResponse = scoreService.expect(member.getId(), LocalDate.now());
        return ResponseEntity.ok(scoreResponse);
    }

    @GetMapping("/api/score/me")
    public ResponseEntity<MyScoreResponse> myScore() {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.PoliceOfficerMale);
        final MyScoreResponse scoreResponse = scoreService.myScores(member.getId());
        return ResponseEntity.ok(scoreResponse);
    }

    @PostMapping("/api/score/me")
    public ResponseEntity<ScoreResponse> updateScore(@RequestBody List<EvaluationScoreRequest> request) {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.PoliceOfficerMale);
        final ScoreResponse scoreResponse = scoreService.update(member.getId(), request, LocalDate.now());
        return ResponseEntity.ok(scoreResponse);
    }

    @GetMapping("/api/score/detail/me")
    public ResponseEntity<List<EvaluationScoreByItemResponse>> evaluationScoresOfItem() {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.PoliceOfficerMale);
        final List<EvaluationScoreByItemResponse> scores = scoreService.evaluationScores(member.getId(), member.getApplicationType(), LocalDate.now());
        return ResponseEntity.ok(scores);
    }

    @GetMapping("/api/score/detail/top")
    public ResponseEntity<List<EvaluationScoreByItemResponse>> rankersEvaluationScoresOfItem(int percent) {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.PoliceOfficerMale);
        final List<EvaluationScoreByItemResponse> scores = scoreService.avgEvaluationItemScoresTopOf(member.getApplicationType(), percent, LocalDate.now());
        return ResponseEntity.ok(scores);
    }

    @GetMapping("/api/score/rank")
    public ResponseEntity<List<RankResponse>> rank(@Valid @NotNull Integer rankCnt) {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.PoliceOfficerMale);
        final List<RankResponse> rankResponses = scoreService.rank(member.getApplicationType(), rankCnt, LocalDate.now());
        return ResponseEntity.ok(rankResponses);
    }
}
