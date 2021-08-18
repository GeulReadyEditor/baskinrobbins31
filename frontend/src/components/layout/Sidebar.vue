<template>
	<div>
		<v-navigation-drawer
			v-model="drawer"
			temporary
			fixed
			fluid style="height: 100vh;"
			app
			>
			<br><br>
			<div class="side-itms" style="padding-left:5%;">
			<v-list-item-avatar size=100>
				<img src="https://randomuser.me/api/portraits/women/82.jpg">
			</v-list-item-avatar>
			</div>
			<div class="side-itms">
			<v-list-item-content>
				<v-list-item-title>User Name</v-list-item-title>
			</v-list-item-content>
			</div>
			<br>
			<div class="side-itms">
			<div v-if="isAuth">
				<v-btn
					class="ma-2"
					rounded
					outlined
					color="#00d5aa">
					글쓰기
				</v-btn>
			</div>
			</div>
			<br>
			<v-divider></v-divider>

			<v-list dense>
				<div>
					<v-list-item
					v-for="item in menusBL"
					:key="item.title">
					<v-list-item-content>
						<v-btn text><v-list-item-title>{{ item.title }}</v-list-item-title></v-btn>
					</v-list-item-content>
					</v-list-item>
				</div>
				<br><br>
				<div v-if="isAuth">
					<v-list-item
					v-for="item in menusAL"
					:key="item.title">
					<v-list-item-content>
						<v-btn text><v-list-item-title>{{ item.title }}</v-list-item-title></v-btn>
					</v-list-item-content>
					</v-list-item>
				</div>
			</v-list>

			<div class="side-itms">
			<div v-if="isAuth" id="side-footer1">
				<v-btn
					class="ma-2"
					rounded
					outlined
					color="#00d5aa"
					@click="logout">
				로그아웃
				</v-btn>
			</div>
			<div v-else id="side-footer2">
			<v-btn
				class="ma-2"
				rounded
				outlined
				color="grey">
				회원가입
			</v-btn>

			<v-btn
				class="ma-2"
				rounded
				outlined
				color="#00d5aa"
				@click="login">
				로그인
			</v-btn>
			</div>
			</div>

		</v-navigation-drawer>
	</div>
</template>

<script lang="ts">
	import Vue from 'vue'

	export default Vue.extend({
		name:"Sidebar",
		data: () => ({
			drawer: false,
			group: null,
			isAuth: false,
			menusBL: [
				{ title: '홈'},
				{ title: '최신글'},
				{ title: '카테고리'},
			],
			menusAL: [
				{ title: '내 프로필'},
				{ title: '내 원고'},
				{ title: '스크랩'},
			],
		}),
		methods:{
			logout(){
				this.isAuth = false;
			},
			login(){
				this.isAuth = true;
			}
		}
	})
</script>

<style>
	.side-itms{
		text-align:center;

	}
	.v-btn::before{
		background-color: transparent !important;
	}
	.v-btn:hover,
	.v-btn:focus,
	.v-btn:active{
		color: #00d5aa !important;
	}
	#side-footer2{
		position:fixed !important;
		bottom:3%;
		left:10%;
	}
	#side-footer1{
		position:fixed !important;
		bottom:3%;
		left:28%;
	}
</style>