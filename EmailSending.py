import smtplib
QQMAIL_USER = '850918164@qq.com'
QQMAIL_PASS = 'jwmhbwzhqrwmbgah'
QQMAIL_PASS1 = 'trjlmyhosmmabdfg'
SMTP_SERVER = 'smtp.qq.com'
SMTP_PORT = 25
def send_email():
    recipient='850918164@qq.com'
    subject = 'Emergency'
    text ='Please Go back to home quickly, Emergency happened!'
    smtpserver = smtplib.SMTP(SMTP_SERVER,SMTP_PORT)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.ehlo
    smtpserver.login(QQMAIL_USER,QQMAIL_PASS1)
    header = 'To:'+recipient+'\n'+'From:'+QQMAIL_USER
    header = header + '\n' +'Subject:' + subject +'\n'
    msg = header +'\n'+text+'\n\n'
    smtpserver.sendmail(QQMAIL_USER,recipient,msg)
    smtpserver.close()
# send_email()