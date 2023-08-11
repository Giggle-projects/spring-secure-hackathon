package se.ton.t210.service.mail;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Profile;
import org.springframework.mail.SimpleMailMessage;
import org.springframework.mail.javamail.JavaMailSender;
import org.springframework.stereotype.Component;
import se.ton.t210.dto.Email;

@Profile("prod")
@Component
public class JavaMailServiceImpl implements MailServiceInterface {

    @Value("${auth.mail.fromAddress:cspft@gmail.com}")
    private String fromAddress;

    private final JavaMailSender mailSender;

    public JavaMailServiceImpl(JavaMailSender mailSender) {
        this.mailSender = mailSender;
    }

    @Override
    public void sendMail(Email email) {
        sendMail(email, fromAddress);
    }

    public void sendMail(Email email, String fromAddress) {
        final SimpleMailMessage mailMessage = new SimpleMailMessage();
        mailMessage.setTo(email.getToAddress());
        mailMessage.setSubject(email.getTitle());
        mailMessage.setText(email.getContent());
        mailMessage.setFrom(fromAddress);
        sendMail(mailMessage);
    }

    public void sendMail(SimpleMailMessage mailMessage) {
        mailSender.send(mailMessage);
    }
}
