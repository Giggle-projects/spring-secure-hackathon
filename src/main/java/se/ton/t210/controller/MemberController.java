package se.ton.t210.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import se.ton.t210.dto.EmailRequest;
import se.ton.t210.dto.ReissuePwdRequest;
import se.ton.t210.dto.SignInRequest;
import se.ton.t210.dto.SignUpRequest;
import se.ton.t210.service.MemberService;

import javax.servlet.http.HttpServletResponse;
import javax.validation.Valid;

@RequestMapping("/api/auth")
@RestController
public class MemberController {

    private final MemberService memberService;

    public MemberController(MemberService memberService) {
        this.memberService = memberService;
    }

    @PostMapping("/signUp")
    public ResponseEntity<Void> signUp(@RequestBody @Valid SignUpRequest request,
                                       HttpServletResponse response) {
        request.validateRequest();
        memberService.signUp(request, response);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/signIn")
    public ResponseEntity<Void> signIn(@RequestBody @Valid SignInRequest request,
                                       HttpServletResponse response) {
        memberService.signIn(request, response);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/reissue/token")
    public ResponseEntity<Void> reissueToken(@CookieValue String accessToken,
                                             @CookieValue String refreshToken,
                                             HttpServletResponse response) {
        memberService.reissueToken(accessToken, refreshToken, response);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/email")
    public ResponseEntity<Void> sendEmailAuthMail(@RequestBody EmailRequest email) {
        memberService.sendEmailAuthMail(email.getEmail());
        return ResponseEntity.ok().build();
    }

    @PostMapping("/reissue/pwd")
    public ResponseEntity<Void> reissuePwd(@RequestBody ReissuePwdRequest request) {
        String email = "devygwan@gmail.com";
        request.validRequest();
        memberService.reissuePwd(email, request.getPassword());
        return ResponseEntity.ok().build();
    }
}
