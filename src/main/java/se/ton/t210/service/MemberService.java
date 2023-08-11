package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import se.ton.t210.cache.EmailAuthCache;
import se.ton.t210.cache.EmailAuthCacheRepository;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.MemberRepository;
import se.ton.t210.dto.Email;
import se.ton.t210.dto.MemberTokens;
import se.ton.t210.dto.SignInRequest;
import se.ton.t210.dto.SignUpRequest;
import se.ton.t210.exception.AuthException;
import se.ton.t210.service.mail.MailServiceInterface;

import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletResponse;
import java.security.SecureRandom;
import java.time.LocalTime;

@Transactional
@Service
public class MemberService {

    @Value("${auth.jwt.token.access.cookie.key:accessToken}")
    private String accessTokenCookieKey;

    @Value("${auth.jwt.token.refresh.cookie:refreshToken}")
    private String refreshTokenCookieKey;

    @Value("${auth.mail.title}")
    private String emailAuthMailTitle;

    @Value("${auth.mail.content.header}")
    private String emailAuthMailContentHeader;

    private final MemberRepository memberRepository;
    private final MemberTokenService memberTokenService;
    private final MailServiceInterface mailServiceInterface;
    private final EmailAuthCacheRepository emailAuthCacheRepository;

    public MemberService(MemberRepository memberRepository, MemberTokenService memberTokenService, MailServiceInterface mailServiceInterface, EmailAuthCacheRepository emailAuthCacheRepository) {
        this.memberRepository = memberRepository;
        this.memberTokenService = memberTokenService;
        this.mailServiceInterface = mailServiceInterface;
        this.emailAuthCacheRepository = emailAuthCacheRepository;
    }

    public void signUp(SignUpRequest request, HttpServletResponse response) {
        if (memberRepository.existsByEmail(request.getEmail())) {
            throw new AuthException(HttpStatus.CONFLICT, "email is already exists");
        }
        final Member member = memberRepository.save(request.toEntity());
        final MemberTokens tokens = memberTokenService.createTokensByEmail(member.getEmail());
        responseTokens(response, tokens);
    }

    public void signIn(SignInRequest request, HttpServletResponse response) {
        if (!memberRepository.existsByEmailAndPassword(request.getEmail(), request.getPassword())) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "The username or password is not valid.");
        }
        final MemberTokens tokens = memberTokenService.createTokensByEmail(request.getEmail());
        responseTokens(response, tokens);
    }

    public void reissueToken(String accessToken, String refreshToken, HttpServletResponse response) {
        final MemberTokens tokens = memberTokenService.reissue(accessToken, refreshToken);
        responseTokens(response, tokens);
    }

    public void sendEmailAuthMail(String userEmail) {
        String authCode = createCode();
        Email email = new Email(emailAuthMailTitle, emailAuthMailContentHeader + authCode, userEmail);
        mailServiceInterface.sendMail(email);
        emailAuthCacheRepository.save(new EmailAuthCache(userEmail, authCode, LocalTime.now()));
    }

    @Transactional
    public void reissuePwd(String email, String newPwd) {
        Member member = memberRepository.findByEmail(email).orElseThrow(() ->
                new AuthException(HttpStatus.NOT_FOUND, "User is not found"));
        if (member.getEmail().equals(newPwd)) {
            throw new IllegalArgumentException("The password you want to change must be different from the previous password.");
        }
        member.reissuePwd(newPwd);
        memberRepository.save(member);
    }

    private String createCode() {
        int length = 10;
        try {
            SecureRandom secureRandom = new SecureRandom();
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < length; i++) {
                builder.append(secureRandom.nextInt(10));
            }
            return builder.toString();
        } catch (Exception e) {
            throw new IllegalArgumentException("email authentication code create error");
        }
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
