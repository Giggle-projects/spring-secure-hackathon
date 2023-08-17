package se.ton.t210.domain;

import javax.persistence.metamodel.SingularAttribute;
import javax.persistence.metamodel.StaticMetamodel;
import java.time.LocalDateTime;

@StaticMetamodel(Member.class)
public class AccessDateTime_ {
    public static volatile SingularAttribute<AccessDateTime, Long> id;
    public static volatile SingularAttribute<AccessDateTime, LocalDateTime> accessTime;
    public static volatile SingularAttribute<AccessDateTime, Long> memberId;
}
