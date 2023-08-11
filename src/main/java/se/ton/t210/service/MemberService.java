package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.MemberRepository;
import se.ton.t210.dto.Email;
import se.ton.t210.dto.LogInRequest;
import se.ton.t210.dto.MemberTokens;
import se.ton.t210.dto.SignUpRequest;
import se.ton.t210.exception.AuthException;
import se.ton.t210.service.mail.MailServiceInterface;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletResponse;

@Transactional
@Service
public class MemberService {

    @Value("${auth.jwt.token.access.cookie.key:accessToken}")
    private String accessTokenCookieKey;

    @Value("${auth.jwt.token.refresh.cookie:refreshToken}")
    private String refreshTokenCookieKey;

    @Value("${auth.mail.title}")
    private String emailAuthMailTitle;

    @Value("${auth.mail.content}")
    private String emailAuthMailContent;

    private final MemberRepository memberRepository;
    private final MemberTokenService memberTokenService;
    private final MailServiceInterface mailServiceInterface;

    public MemberService(MemberRepository memberRepository, MemberTokenService memberTokenService, MailServiceInterface mailServiceInterface) {
        this.memberRepository = memberRepository;
        this.memberTokenService = memberTokenService;
        this.mailServiceInterface = mailServiceInterface;
    }

    public void signUp(SignUpRequest request, HttpServletResponse response) {
        if (memberRepository.existsByUsername(request.getUsername())) {
            throw new AuthException(HttpStatus.CONFLICT, "username is already exists");
        }
        final Member member = memberRepository.save(request.toEntity());
        final MemberTokens tokens = memberTokenService.createTokensByUsername(member.getUsername());
        responseTokens(response, tokens);
    }

    public void signIn(LogInRequest request, HttpServletResponse response) {
        if (!memberRepository.existsByUsernameAndPassword(request.getUsername(), request.getPassword())) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "The username or password is not valid.");
        }
        final MemberTokens tokens = memberTokenService.createTokensByUsername(request.getUsername());
        responseTokens(response, tokens);
    }

    public void reissueToken(String accessToken, String refreshToken, HttpServletResponse response) {
        final MemberTokens tokens = memberTokenService.reissue(accessToken, refreshToken);
        responseTokens(response, tokens);
    }

    public void sendMail(String userEmail) {
        Email email = new Email(emailAuthMailTitle, emailAuthMailContent, userEmail);
        mailServiceInterface.sendMail(email);
    }

    private void responseTokens(HttpServletResponse response, MemberTokens tokens) {
        loadTokenCookie(response, accessTokenCookieKey, tokens.getAccessToken());
        loadTokenCookie(response, refreshTokenCookieKey, tokens.getRefreshToken());
    }

    private void loadTokenCookie(HttpServletResponse response, String key, String value) {
        final Cookie cookie = new Cookie(key, value);
        cookie.setHttpOnly(true);
        cookie.setPath("/");
        response.addCookie(cookie);
    }
}
