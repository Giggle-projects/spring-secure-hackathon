package se.ton.t210.service.mail;

import se.ton.t210.dto.Email;

public interface MailServiceInterface {

    void sendMail(Email email);

    void sendMail(Email email, String fromAddress);
}