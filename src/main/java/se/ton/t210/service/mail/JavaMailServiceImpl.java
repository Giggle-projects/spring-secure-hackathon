package se.ton.t210.service.mail;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Component;
import se.ton.t210.service.mail.form.MailForm;

@ConditionalOnProperty(value = "auth.mail.enable.mode", havingValue = "true")
@Async
@Component
public class JavaMailServiceImpl implements MailServiceInterface {

    @Value("${auth.mail.fromAddr:cspft@gmail.com}")
    private String fromAddr;

    private final JavaMailSender mailSender;

    public JavaMailServiceImpl(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }

    @Override
    public void sendMail(String userAddr, MailForm form, String fromAddr) {
        final SimpleMailMessage mailMessage = new SimpleMailMessage();
        mailMessage.setTo(userAddr);
        mailMessage.setSubject(form.title());
        mailMessage.setText(form.body());
        mailMessage.setFrom(fromAddr);
        sendMail(mailMessage);
    }

    public void sendMail(String userAddr, MailForm form) {
        sendMail(userAddr, form, fromAddr);
    }

    public void sendMail(SimpleMailMessage mailMessage) {
        mailSender.send(mailMessage);
    }
}
