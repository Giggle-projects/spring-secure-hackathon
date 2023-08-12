package se.ton.t210.service.mail.form;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

@Component
public class AuthEmailForm {

    @Value("${auth.mail.title}")
    private String emailAuthMailTitle;

    @Value("${auth.mail.content.header}")
    private String emailAuthMailContentHeader;

    public Email creatEmail(String content, String toAddress) {
        return new Email(emailAuthMailTitle, emailAuthMailContentHeader + content, toAddress);
    }
}
