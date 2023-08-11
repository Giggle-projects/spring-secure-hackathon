package se.ton.t210.cache;

import org.springframework.data.repository.CrudRepository;

public interface EmailAuthCacheRepository extends CrudRepository<EmailAuthCache, String> {
}
