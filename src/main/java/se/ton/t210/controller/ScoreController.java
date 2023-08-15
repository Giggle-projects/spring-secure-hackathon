package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
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
    public ResponseEntity<RecordCountResponse> recordCount() {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.FireOfficerFemale);
        final RecordCountResponse countResponse = scoreService.count(member.getApplicationType());
        return ResponseEntity.ok(countResponse);
    }

    @GetMapping("/api/score")
    public ResponseEntity<ScoreResponse> evaluationScore(@Valid @NotNull Long evaluationItemId,
                                                         @Valid @NotNull Integer score) {
        final int evaluationScore = scoreService.evaluationScore(evaluationItemId, score);
        return ResponseEntity.ok(new ScoreResponse(evaluationScore));
    }

    @GetMapping("/api/score/year")
    public ResponseEntity<List<MonthlyScoreResponse>> scores() {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.FireOfficerFemale);
        final List<MonthlyScoreResponse> scoresYear = scoreService.scoresYear(member, LocalDate.now());
        return ResponseEntity.ok(scoresYear);
    }

    @GetMapping("/api/score/expect")
    public ResponseEntity<ExpectScoreResponse> expect() {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.FireOfficerFemale);
        final ExpectScoreResponse scoreResponse = scoreService.expect(member.getId(), LocalDate.now());
        return ResponseEntity.ok(scoreResponse);
    }

    @GetMapping("/api/score/me")
    public ResponseEntity<ScoreResponse> score() {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.FireOfficerFemale);
        final ScoreResponse scoreResponse = scoreService.score(member.getId());
        return ResponseEntity.ok(scoreResponse);
    }

    @PostMapping("/api/score/me")
    public ResponseEntity<ScoreResponse> updateScore(@RequestBody List<EvaluationScoreRequest> request) {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.FireOfficerFemale);
        final ScoreResponse scoreResponse = scoreService.update(member.getId(), request, LocalDate.now());
        return ResponseEntity.ok(scoreResponse);
    }

    @GetMapping("/api/score/rank")
    public ResponseEntity<List<RankResponse>> rank(@Valid @NotNull Integer rankCnt) {
        final Member member = new Member(1l, "홍길동", "email", "password", ApplicationType.FireOfficerFemale);
        final List<RankResponse> rankResponses = scoreService.rank(member.getApplicationType(), rankCnt, LocalDate.now());
        return ResponseEntity.ok(rankResponses);
    }
}
