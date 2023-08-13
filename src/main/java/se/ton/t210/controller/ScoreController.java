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

//    @GetMapping("/api/application/rank")
//    public ResponseEntity<List<RankResponse>> rank() {
//        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
//        final List<RankResponse> rankResponses = scoreService.rank(member.getApplicationType(), 5);
//        return ResponseEntity.ok(rankResponses);
//    }
//
//    @GetMapping("/api/records/count")
//    public ResponseEntity<RecordCountResponse> recordCount() {
//        final Member member = new Member(1l, "name", "email", "password", Gender.MALE, ApplicationType.FireOfficerFemale);
//        final RecordCountResponse countResponse = scoreService.count(member.getApplicationType());
//        return ResponseEntity.ok(countResponse);
//    }
}
