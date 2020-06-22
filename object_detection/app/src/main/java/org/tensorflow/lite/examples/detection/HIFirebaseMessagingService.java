package org.tensorflow.lite.examples.detection;

import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.RingtoneManager;
import android.net.Uri;
import android.os.Build;

import androidx.core.app.NotificationCompat;

import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;

import org.tensorflow.lite.examples.detection.env.Logger;

import java.io.IOException;
import java.net.URL;

public class HIFirebaseMessagingService extends FirebaseMessagingService {
    private static final Logger LOGGER = new Logger();

    @Override
    public void onNewToken(String token) {
        LOGGER.d("FCM New Token : " + token);
    }

    @Override
    public void onMessageReceived(RemoteMessage remoteMsg) {
        if (remoteMsg.getNotification() != null) {
            LOGGER.d("FCM Notification : " + remoteMsg.getNotification());
            LOGGER.d("FCM Notification Message : " + remoteMsg.getNotification().getBody());
            String msgBody = remoteMsg.getNotification().getBody();
            String msgTitle = remoteMsg.getNotification().getTitle();
            Bitmap bitmap = null;
            try {
                LOGGER.d("FCM Image URL : " + remoteMsg.getNotification().getImageUrl());
                if (remoteMsg.getNotification().getImageUrl() != null) {
                    URL img = new URL(remoteMsg.getNotification().getImageUrl().toString());
                    bitmap = BitmapFactory.decodeStream(img.openConnection().getInputStream());
                }
            } catch (IOException e) {
                e.printStackTrace();
                LOGGER.d("FCM Image Exception : " + e);
            }

            Intent intent = new Intent(this, DetectorActivity.class);
            PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent, PendingIntent.FLAG_ONE_SHOT);
            String channelId = "BFER_noti";
            Uri defaultSoundUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION);

            NotificationCompat.Builder notiBuilder =
                    new NotificationCompat.Builder(this, channelId)
                            .setSmallIcon(R.mipmap.ic_launcher)
                            .setContentTitle(msgTitle)
                            .setContentText(msgBody)
                            .setAutoCancel(true)
                            .setSound(defaultSoundUri)
                            .setContentIntent(pendingIntent);

            if (bitmap != null) {
                notiBuilder.setStyle(new NotificationCompat.BigPictureStyle().bigPicture(bitmap));
            }

            NotificationManager notiManager = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                String channelName = "BFER_Noti";
                NotificationChannel channel = new NotificationChannel(channelId, channelName, NotificationManager.IMPORTANCE_HIGH);
                notiManager.createNotificationChannel(channel);
            }
            notiManager.notify(0, notiBuilder.build());
        }
    }

    private void sendNotification(String msgBody) {

    }
}
