package se.ton.t210.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import se.ton.t210.cache.EmailAuthMailCache;
import se.ton.t210.cache.EmailAuthMailCacheRepository;
import se.ton.t210.exception.AuthException;

import java.security.SecureRandom;
import java.time.Duration;
import java.time.LocalTime;

@Service
public class MailAuthService {

    @Value("${auth.mail.valid.time}")
    private Long mailValidTime;

    @Value("${auth.code.length}")
    private int authCodeLength;

    private final EmailAuthMailCacheRepository emailAuthMailCacheRepository;

    public MailAuthService(EmailAuthMailCacheRepository emailAuthMailCacheRepository) {
        this.emailAuthMailCacheRepository = emailAuthMailCacheRepository;
    }

    public void saveAuthInfoFromEmailAuthMailCache(String userEmail, String authCode) {
        emailAuthMailCacheRepository.save(new EmailAuthMailCache(userEmail, authCode, LocalTime.now()));
    }

    public String createAuthCode() {
        try {
            SecureRandom secureRandom = new SecureRandom();
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < authCodeLength; i++) {
                builder.append(secureRandom.nextInt(10));
            }
            return builder.toString();
        } catch (Exception e) {
            throw new IllegalArgumentException("email authentication code create error");
        }
    }

    public void validateAuthCode(String email, String authCode) {
        EmailAuthMailCache emailAuthMailCache = emailAuthMailCacheRepository.findById(email).orElseThrow(() ->
                new AuthException(HttpStatus.NOT_FOUND, "Email not found"));
        long afterSeconds = Duration.between(emailAuthMailCache.getCreateTime(), LocalTime.now()).getSeconds();
        if (afterSeconds > mailValidTime) {
            throw new AuthException(HttpStatus.REQUEST_TIMEOUT, "Email valid time is exceed");
        }
        if (!emailAuthMailCache.getAuthCode().equals(authCode)) {
            throw new AuthException(HttpStatus.UNAUTHORIZED, "AuthCode is not correct");
        }
    }
}
