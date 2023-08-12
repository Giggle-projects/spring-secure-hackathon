package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import se.ton.t210.cache.EmailAuthMailCache;
import se.ton.t210.cache.EmailAuthMailCacheRepository;
import se.ton.t210.exception.AuthException;
import se.ton.t210.service.mail.MailServiceInterface;
import se.ton.t210.service.mail.form.SignUpAuthMailForm;
import se.ton.t210.service.token.AuthTokenService;
import se.ton.t210.utils.http.CookieUtils;

import javax.servlet.http.HttpServletResponse;
import java.time.Duration;
import java.time.LocalTime;

@Service
public class AuthService {

    @Value("${auth.mail.valid.time}")
    private Long mailValidTime;

    @Value("${auth.code.length}")
    private int authCodeLength;

    @Value("${auth.jwt.token.email.cookie.key:emailAuthToken}")
    private String authTokenCookieKey;

    private final EmailAuthMailCacheRepository emailAuthMailCacheRepository;
    private final AuthTokenService authTokenService;
    private final MailServiceInterface mailServiceInterface;

    public AuthService(EmailAuthMailCacheRepository emailAuthMailCacheRepository, AuthTokenService authTokenService, MailServiceInterface mailServiceInterface) {
        this.emailAuthMailCacheRepository = emailAuthMailCacheRepository;
        this.authTokenService = authTokenService;
        this.mailServiceInterface = mailServiceInterface;
    }

    public void sendEmailAuthMail(String userEmailAddress) {
        String authCode = AuthCodeUtils.createAuthNumberCode(authCodeLength);
        mailServiceInterface.sendMail(userEmailAddress, new SignUpAuthMailForm(authCode));
        saveAuthInfoFromEmailAuthMailCache(userEmailAddress, authCode);
    }

    private void saveAuthInfoFromEmailAuthMailCache(String userEmail, String authCode) {
        emailAuthMailCacheRepository.save(new EmailAuthMailCache(userEmail, authCode, LocalTime.now()));
    }

    public void validateAuthCode(String email, String authCode, HttpServletResponse response) {
        EmailAuthMailCache emailAuthMailCache = emailAuthMailCacheRepository.findById(email).orElseThrow(() ->
                new AuthException(HttpStatus.NOT_FOUND, "Email not found"));
        long afterSeconds = Duration.between(emailAuthMailCache.getCreatedTime(), LocalTime.now()).getSeconds();
        if (afterSeconds > mailValidTime) {
            throw new AuthException(HttpStatus.REQUEST_TIMEOUT, "Email valid time is exceed");
        }
        if (!emailAuthMailCache.getAuthCode().equals(authCode)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "AuthCode is not correct");
        }
        if (!emailAuthMailCache.getEmail().equals(email)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "email is not correct");
        }
        String authToken = authTokenService.createAuthTokenByEmail(email);
        CookieUtils.loadHttpOnlyCookie(response, authTokenCookieKey, authToken);
    }
}
