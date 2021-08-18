<template>
	<v-footer
		dark
		padless
		app
		bottom
		absolute
		inset
		height="200"
	>
    <v-card id="footer-padding"
      flat
      tile
      class="grey darken-3 white--text text-center"
    >
      <v-card-text id="footer-icons">
        <v-btn
          v-for="icon in icons"
          :key="icon"
          class="mx-4 white--text"
          icon
        >
          <v-icon size="24px">
            {{ icon }}
          </v-icon>
        </v-btn>
      </v-card-text>

      <v-card-text class="mx-auto white--text">
		  <div class="footer-paragraph"><a href="#" class='footer-content'>원고소개</a> |
			<a href="#" class='footer-content'>이용약관</a> |
			<a href="#" class='footer-content'>개인정보처리방침</a> |
			<a href="#" class='footer-content'>운영정책</a> |
			<a href="#" class='footer-content'>고객센터</a>
		  </div>
      </v-card-text>
      <div align="center"><v-divider class="divider-footer"></v-divider></div>
      <v-card-text class="white--text">
        <img src="@/assets/logo/onego_white.png" width="100px" height="36px"><br>
		<em style="opacity:60%;">© Copyright 2021. All rights reserved.</em>
      </v-card-text>
    </v-card>
  </v-footer>
</template>

<script lang="ts">
	import Vue from 'vue'

	export default Vue.extend({
		data: () => ({
			icons: [
				'mdi-facebook',
				'mdi-twitter',
				'mdi-linkedin',
				'mdi-instagram',
			],
		}),
	})
</script>

<style>
	#footer-padding{
		width:100%;
	}
	.divider-footer{
		text-align: center;
		width:55%;
	}
	#footer-icons{
		margin-bottom:-10px;
	}
	.footer-paragraph{
		word-spacing: 1.2em;
	}
	.footer-content{
		text-decoration: none !important;
		color: white !important;
	}
	.footer-content:hover{
		color:#00d5aa !important;
		text-decoration: underline !important;
	}
  
</style>